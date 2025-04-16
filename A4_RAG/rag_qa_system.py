#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pypdf')


# In[2]:


get_ipython().system('pip install -U langchain-community')


# In[3]:


get_ipython().system('pip install rank_bm25')


# In[4]:


# Install required packages
get_ipython().system('pip install rank_bm25 faiss-gpu langchain langchain-community langchain-core transformers sentence-transformers ragas')

from huggingface_hub import login
login(token="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")

import os
import nltk
import pandas as pd
import textwrap
from typing import List, Tuple, Dict

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# BM25 imports
from rank_bm25 import BM25Okapi

# Model imports
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Initialize NLTK (for tokenization if needed)
nltk.download('punkt')

# Constants
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_SEARCH_K = 3
DEFAULT_SEARCH_TYPE = "hybrid" # 'semantic', 'keyword', 'hybrid'
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "meta-llama/Llama-3.2-1B"


# # Document Processing Functions

# In[5]:


def load_documents(directory: str, glob_pattern: str = "**/*.pdf") -> List[Document]:
    """Load documents from a directory using PyPDFLoader."""
    loader = DirectoryLoader(directory, glob=glob_pattern, loader_cls=PyPDFLoader)
    return loader.load()

def chunk_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    separators: List[str] = None
) -> List[Document]:
    """Split documents into chunks using specified parameters."""
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )
    return text_splitter.split_documents(documents)

def create_vector_store(
    chunks: List[Document],
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    save_path: str = None
) -> FAISS:
    """Create and optionally save a FAISS vector store."""
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectordb = FAISS.from_documents(chunks, embeddings)

    if save_path:
        vectordb.save_local(save_path)
    return vectordb

def create_bm25_index(chunks: List[Document]) -> BM25Okapi:
    """Create a BM25 index from document chunks."""
    texts = [chunk.page_content for chunk in chunks]
    tokenized_texts = [text.split() for text in texts]
    return BM25Okapi(tokenized_texts)


# # Search Functions

# In[6]:


def semantic_search(
    query: str,
    vectordb: FAISS,
    k: int = DEFAULT_SEARCH_K,
    score_threshold: float = None
) -> List[Tuple[Document, float]]:
    """
    Perform semantic search using vector similarity.

    Args:
        query: Search query
        vectordb: FAISS vector store
        k: Number of results to return
        score_threshold: Minimum similarity score (None for no threshold)

    Returns:
        List of (Document, score) tuples
    """
    results = vectordb.similarity_search_with_score(query, k=k)

    if score_threshold is not None:
        results = [(doc, score) for doc, score in results if score >= score_threshold]

    return results

def keyword_search(
    query: str,
    bm25_index: BM25Okapi,
    chunks: List[Document],
    k: int = DEFAULT_SEARCH_K,
    score_threshold: float = None
) -> List[Tuple[Document, float]]:
    """
    Perform keyword search using BM25.

    Args:
        query: Search query
        bm25_index: BM25 index
        chunks: Original document chunks (for metadata)
        k: Number of results to return
        score_threshold: Minimum BM25 score (None for no threshold)

    Returns:
        List of (Document, score) tuples
    """
    tokenized_query = query.split()
    scores = bm25_index.get_scores(tokenized_query)

    # Get top-k indices
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    # Create results with documents and scores
    results = [(chunks[i], scores[i]) for i in top_k_indices]

    if score_threshold is not None:
        results = [(doc, score) for doc, score in results if score >= score_threshold]

    return results

def hybrid_search(
    query: str,
    vectordb: FAISS,
    bm25_index: BM25Okapi,
    chunks: List[Document],
    k: int = DEFAULT_SEARCH_K,
    semantic_weight: float = 0.5,
    keyword_weight: float = 0.5
) -> List[Tuple[Document, float]]:
    """
    Perform hybrid search combining semantic and keyword approaches.

    Args:
        query: Search query
        vectordb: FAISS vector store
        bm25_index: BM25 index
        chunks: Original document chunks
        k: Number of results to return
        semantic_weight: Weight for semantic search scores
        keyword_weight: Weight for keyword search scores

    Returns:
        List of (Document, combined_score) tuples
    """
    # Get semantic results
    semantic_results = semantic_search(query, vectordb, k*2)
    semantic_scores = {doc.page_content: score for doc, score in semantic_results}

    # Get keyword results
    keyword_results = keyword_search(query, bm25_index, chunks, k*2)
    keyword_scores = {doc.page_content: score for doc, score in keyword_results}

    # Combine scores
    all_docs = set(semantic_scores.keys()).union(set(keyword_scores.keys()))
    combined_scores = []

    for doc_content in all_docs:
        sem_score = semantic_scores.get(doc_content, 0)
        kw_score = keyword_scores.get(doc_content, 0)

        # Normalize scores (BM25 can be unbounded, so we normalize to 0-1 range)
        max_kw_score = max(keyword_scores.values()) if keyword_scores else 1
        norm_kw_score = kw_score / max_kw_score if max_kw_score > 0 else 0

        # Combine with weights
        combined_score = (semantic_weight * sem_score) + (keyword_weight * norm_kw_score)
        combined_scores.append((doc_content, combined_score))

    # Sort by combined score and get top-k
    combined_scores.sort(key=lambda x: x[1], reverse=True)
    top_scores = combined_scores[:k]

    # Map back to documents
    doc_lookup = {chunk.page_content: chunk for chunk in chunks}
    results = []

    for doc_content, score in top_scores:
        if doc_content in doc_lookup:
            results.append((doc_lookup[doc_content], score))

    return results


# # LLM functions

# In[48]:


def initialize_llm(
    model_name: str = DEFAULT_LLM_MODEL,
    device: str = "cuda",
    max_new_tokens: int = 300
) -> Tuple[pipeline, any]:
    """
    Initialize the LLM pipeline.

    Returns:
        Tuple of (generator, tokenizer)
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.chat_template = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}User: {{ message['content'] }}\n"
    "{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}Assistant:{% endif %}"
)


    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        max_new_tokens=max_new_tokens,
        do_sample=True
    )

    return generator, tokenizer

def generate_response(
    prompt: str,
    generator: pipeline,
    width: int = 80
) -> str:
    """Generate a response from the LLM."""
    messages = [{"role": "user", "content": prompt}]
    output = generator(messages)
    return textwrap.fill(output[0]["generated_text"], width=width)

def format_rag_prompt(
    question: str,
    retrieved_docs: List[Document],
    instruction: str = None
) -> str:
    """
    Format a RAG prompt with retrieved documents.

    Args:
        question: User question
        retrieved_docs: List of retrieved documents
        instruction: Custom instructions for the LLM

    Returns:
        Formatted prompt string
    """
    if instruction is None:
        instruction = """You are an AI assistant tasked with answering questions based on retrieved knowledge.
- Integrate the key points from all retrieved responses into a cohesive, well-structured answer.
- If the responses are contradictory, mention the different perspectives.
- If none of the retrieved responses contain relevant information, reply:
  "I couldn't find a good response to your query in the database."
"""

    retrieved_info = "\n\n".join(
        f"{i+1}️⃣ {doc.page_content[:1000]}..." if len(doc.page_content) > 1000
        else f"{i+1}️⃣ {doc.page_content}"
        for i, doc in enumerate(retrieved_docs)
    )

    return f"""
{instruction}

### Retrieved Information:
{retrieved_info}

### Question:
{question}
"""


# # RAG Evaluator

# In[8]:


get_ipython().system('pip install ragas')


# In[95]:


get_ipython().system('pip install datasets matplotlib')

from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextRecall,
    ContextPrecision,
    AnswerCorrectness
)
from datasets import Dataset
import matplotlib.pyplot as plt
import pandas as pd
from langchain.llms import HuggingFaceHub

class HuggingFaceLLM:
    def __init__(self, model_name: str):
        # Initialize Hugging Face model using langchain
        self.model = HuggingFaceHub(repo_id=model_name)

    def __call__(self, prompt: str) -> str:
        # Call the model with the given prompt and return the output
        return self.model(prompt)

class RAGEvaluator:
    def __init__(self, pipeline, llm, embeddings):
        self.pipeline = pipeline
        self.embeddings = embeddings

        self.results = []

        # Make sure llm is a proper LLM object, not a string
        if isinstance(llm, str):
            # Convert from string to LLM object
            from langchain.llms import HuggingFaceHub
            self.llm = HuggingFaceHub(repo_id=llm)
        else:
            # Use the provided LLM object
            self.llm = llm

        # If no LLM provided, create a default one
        if self.llm is None:
            from langchain.llms import HuggingFaceHub
            self.llm = HuggingFaceHub(repo_id="meta-llama/Llama-3.2-1B")

    def evaluate_ragas(self, questions: list, gold_answers: list = None):
        """
        Full RAG evaluation with Ragas metrics
        Args:
            questions: List of questions to evaluate
            gold_answers: Optional list of reference answers for answer_correctness
        Returns:
            DataFrame with complete metrics
        """

        rag_results = []

        for question, gold_answer in zip(questions, gold_answers or [None]*len(questions)):
            # Get RAG response
            answer = self.pipeline.query(question)
            contexts = [doc.page_content for doc in self.pipeline.get_last_retrieved_docs()]

            # Prepare dataset for Ragas
            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts]
            }
            if gold_answer:
                data["ground_truth"] = [gold_answer]

            # Evaluate with Ragas
            metrics = [Faithfulness(), AnswerRelevancy(), ContextRecall(), ContextPrecision()]
            if gold_answer:
                metrics.append(AnswerCorrectness())

            result = evaluate(Dataset.from_dict(data), metrics=metrics, llm=self.llm, embeddings=self.embeddings)
            print(result)

        return result

    def visualize_metrics(self):
        """Generate visualization of evaluation metrics"""
        if self.results.empty:
            raise ValueError("Run evaluate_ragas() first")

        metrics = ['Faithfulness', 'AnswerRelevancy', 'ContextRecall', 'ContextPrecision']
        if 'answer_correctness' in self.results.columns:
            metrics.append('AnswerCorrectness')

        plt.figure(figsize=(10, 6))
        self.results[metrics].mean().plot(kind='bar', color='skyblue')
        plt.title('RAG Performance Metrics')
        plt.ylabel('Score (0-1)')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.show()

        # Context analysis
        plt.figure(figsize=(8, 4))
        self.results['retrieved_docs'].value_counts().sort_index().plot(kind='bar', color='lightgreen')
        plt.title('Number of Retrieved Contexts per Query')
        plt.xlabel('Number of Documents')
        plt.ylabel('Count')
        plt.show()

    def get_optimization_insights(self):
        """Generate actionable insights for RAG optimization"""
        insights = []
        df = self.results

        # Faithfulness issues
        if df['Faithfulness'].mean() < 0.7:
            insights.append("Low faithfulness scores indicate hallucinations or missing context")
            insights.append("Try increasing chunk overlap or adding metadata filtering")

        # Context recall issues
        if df['ContextRecall'].mean() < 0.6:
            insights.append("Low context recall means missing relevant documents")
            insights.append("Try different chunk sizes or hybrid search approaches")

        # Context precision issues
        if df['ContextPrecision'].mean() < 0.6:
            insights.append("Low context precision means too many irrelevant documents")
            insights.append("Improve retrieval with better embeddings or query expansion")

        # Answer relevance issues
        if df['AnswerRelevance'].mean() < 0.7:
            insights.append("Answers don't match questions well")
            insights.append("Improve prompt engineering or add query understanding")

        return "\n".join(insights)


# # RAG Pipeline

# In[28]:


class RAGPipeline:
    def __init__(
        self,
        document_dir: str,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        llm_model: str = DEFAULT_LLM_MODEL,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        device: str = "cuda"
    ):
        """Initialize the RAG pipeline."""
        self.document_dir = document_dir
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.device = device

        # Initialize components
        self.documents = None
        self.chunks = None
        self.vectordb = None
        self.bm25_index = None
        self.llm = None
        self.tokenizer = None
        self.last_retrieved_docs = None  # Track last retrieved documents


    def load_and_process_documents(self):
        """Load and process documents into chunks."""
        print("Loading documents...")
        self.documents = load_documents(self.document_dir)
        print(f"Loaded {len(self.documents)} pages")

        print("Chunking documents...")
        self.chunks = chunk_documents(
            self.documents,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        print(f"Created {len(self.chunks)} chunks")

        # Add IDs to chunks
        for i, chunk in enumerate(self.chunks):
            chunk.metadata["chunk_id"] = i

    def initialize_retrieval(self):
        """Initialize retrieval systems (vector store and BM25)."""
        if not self.chunks:
            raise ValueError("Documents must be loaded and chunked first")

        print("Creating vector store...")
        self.vectordb = create_vector_store(self.chunks, self.embedding_model)

        print("Creating BM25 index...")
        self.bm25_index = create_bm25_index(self.chunks)

    def initialize_llm(self):
        """Initialize the LLM."""
        print("Initializing LLM...")
        self.llm, self.tokenizer = initialize_llm(self.llm_model, self.device)


    def get_last_retrieved_docs(self):
        """Get the documents retrieved in the last query"""
        if self.last_retrieved_docs is None:
            raise ValueError("No documents have been retrieved yet")
        return self.last_retrieved_docs

    def query(
        self,
        question: str,
        search_type: str = DEFAULT_SEARCH_TYPE,  # "semantic", "keyword", or "hybrid"
        k: int = DEFAULT_SEARCH_K,
        semantic_weight: float = 0.5,
        keyword_weight: float = 0.5,
        custom_instruction: str = None
    ) -> str:
        """
        Execute a full RAG pipeline query.

        Args:
            question: The question to answer
            search_type: Type of search ("semantic", "keyword", or "hybrid")
            k: Number of documents to retrieve
            semantic_weight: Weight for semantic search (hybrid only)
            keyword_weight: Weight for keyword search (hybrid only)
            custom_instruction: Custom LLM instruction

        Returns:
            Generated answer
        """
        if not self.vectordb or not self.bm25_index:
            raise ValueError("Retrieval systems not initialized")
        if not self.llm:
            raise ValueError("LLM not initialized")

        # Retrieve documents
        if search_type == "semantic":
            results = semantic_search(question, self.vectordb, k)
            retrieved_docs = [doc for doc, score in results]
        elif search_type == "keyword":
            results = keyword_search(question, self.bm25_index, self.chunks, k)
            retrieved_docs = [doc for doc, score in results]
        elif search_type == "hybrid":
            results = hybrid_search(
                question,
                self.vectordb,
                self.bm25_index,
                self.chunks,
                k,
                semantic_weight,
                keyword_weight
            )
            retrieved_docs = [doc for doc, score in results]
        else:
            raise ValueError(f"Unknown search type: {search_type}")

        # Format prompt
        prompt = format_rag_prompt(question, retrieved_docs, custom_instruction)
        self.last_retrieved_docs = retrieved_docs

        # Generate response
        return generate_response(prompt, self.llm)

    def experiment(
        self,
        questions: List[str],
        gold_answers: List[str],
        chunk_sizes: List[int],
        k_values: List[int],
        search_types: List[str],
        chunk_overlaps: List[int] = [0, 100, 200],
    ) -> Dict:
        """
        Run experiments with different parameters.

        Returns:
            Dictionary of results for each configuration
        """
        results = {}

        for chunk_size in chunk_sizes:
            for chunk_overlap in chunk_overlaps:
                # Re-chunk documents
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
                self.load_and_process_documents()
                self.initialize_retrieval()
                evaluator = RAGEvaluator(self)

                for search_type in search_types:
                    for k in k_values:
                        config_name = f"chunk{chunk_size}_overlap{chunk_overlap}_{search_type}_k{k}"
                        results[config_name] = {}

                        for question in questions:
                            try:
                                answer = self.query(
                                    question,
                                    search_type=search_type,
                                    k=k
                                )
                                results[config_name][question] = answer
                                evaluation_results = evaluator.evaluate_ragas(questions, gold_answers)

                                # View results
                                print(results[['question', 'faithfulness', 'answer_relevance']].head())

                                # Visualize
                                evaluator.visualize_metrics()

                                # Get optimization tips
                                print("\nOptimization Insights:")
                                print(evaluator.get_optimization_insights())
                            except Exception as e:
                                results[config_name][question] = f"Error: {str(e)}"


        return results


# In[49]:


# Initialize pipeline
rag = RAGPipeline(
    document_dir="/content/sample_data/corpus/",
    embedding_model=DEFAULT_EMBEDDING_MODEL,
    llm_model=DEFAULT_LLM_MODEL,
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP
)



# In[85]:


from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name='distilbert-base-uncased')

# Create a proper LLM instance using your HuggingFaceLLM class
generator, tokenizer = initialize_llm(
    model_name=DEFAULT_LLM_MODEL,
    device="cuda",
    max_new_tokens=300
)

# Create a LangChain wrapper around your existing pipeline
local_llm = HuggingFacePipeline(pipeline=generator)


# In[96]:


# Now use this local_llm with your RAGEvaluator
evaluator = RAGEvaluator(rag, llm=local_llm, embeddings=embeddings)


# In[50]:


# Load and process documents
rag.load_and_process_documents()


# In[51]:


# Initialize retrieval systems and LLM
rag.initialize_retrieval()


# In[54]:


rag.initialize_llm()


# In[97]:


questions = [
    "What is Dynamic Programming?"
#"Explain the matrix method in hashing",
 #   "What are the key concepts in amortized analysis?"
]

# Run single query
answer = rag.query(
    questions[0],
    search_type="hybrid",
    k=3
)
print(answer)
gold_answers = [
    "Dynamic Programming is a powerful technique that can be used to solve many combinatorial problems in polynomial time for which a naive approach would take exponential time. Dynamic Programming is a general approach to solving problems, much like “divide-and-conquer”, except that the subproblems will overlap.",
    #"The matrix method in hashing uses matrix multiplication...",
    #"Amortized analysis is a method for analyzing the average cost of operations over a sequence of operations..."
]

# Run evaluation
results = evaluator.evaluate_ragas(questions, gold_answers)


# In[ ]:


experiment_results = rag.experiment(
    questions=questions,
    gold_answers=gold_answers,
    chunk_sizes=[800, 1000],
    chunk_overlaps=[100, 200],
    search_types=["semantic", "hybrid"],
    k_values=[3, 5] )

# Print experiment results
for config, answers in experiment_results.items():
    print(f"\nConfiguration: {config}")
    for question, answer in answers.items():
        print(f"\nQ: {question}")
        print(f"A: {answer[:200]}...")  # Print first 200 chars of answer

