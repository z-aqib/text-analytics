#!/usr/bin/env python
# coding: utf-8

# ### Acknowledgement
# Name: Hamna Inam, Zara Masood, Zuha Aqib     
# ERP ID: X, Y, 26106    
# Section: 10am Miss Solat    
# Date: 16-Apr-25   

# In[1]:


from datetime import datetime
print("Last time code executed:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


# In[2]:


def print_date_time():
    return "", datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# # Imports
# here we add all imports and prerequisities like installations, authentications, constant definitions etc

# ## Installations
# first we need to install related packages

# In[3]:


# # adding this because kaggle ke maslay
# !pip uninstall -y langchain langchain-core langchain-community langchain-openai ragas pydantic -y


# In[4]:


get_ipython().system('pip install transformers')
get_ipython().system('pip install sentence-transformers')
get_ipython().system('pip install pypdf')
get_ipython().system('pip install pymupdf')
get_ipython().system('pip install rank_bm25')
get_ipython().system('pip install datasets')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install faiss')
get_ipython().system('pip install faiss-cpu')
get_ipython().system('pip install faiss-gpu')
get_ipython().system('pip install --upgrade pypdf')

# 2. Then install
# !pip install "langchain==0.2.0"
# !pip install "langchain-core==0.2.0"
# !pip install "langchain-community==0.2.0"
# !pip install "langchain-text-splitters==0.2.1"
# !pip install "langchain-openai==0.1.0"
# !pip install "pydantic==2.6.4"
# !pip install "ragas==0.2.14"

get_ipython().system('pip install langchain')
get_ipython().system('pip install langchain-core')
get_ipython().system('pip install langchain-community')
get_ipython().system('pip install langchain-text-splitters')
get_ipython().system('pip install langchain-openai')
get_ipython().system('pip install pydantic')
get_ipython().system('pip install ragas')

get_ipython().system('pip install --upgrade pandas')


# In[5]:


get_ipython().system('pip install numpy==1.26.4 scipy==1.12.0 --force-reinstall')


# ## Imports
# here we import the necessary libraries and modules

# In[6]:


# ===== Environment & Authentication =====
import os
import time
import csv
# from dotenv import load_dotenv, dotenv_values
from huggingface_hub import login

# ===== Core Python & Data Handling =====
from typing import List, Tuple, Dict
import textwrap
import pandas as pd
import matplotlib.pyplot as plt

# ===== NLP Preprocessing =====
import nltk
from rank_bm25 import BM25Okapi  # BM25 retriever
nltk.download('punkt')  # Ensure NLTK data is available

# ===== LangChain - Document Loading & Splitting =====
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
# from langchain.schema import Document
from langchain_core.documents import Document

# ===== LangChain - Embeddings & Vector Stores =====
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# ===== Hugging Face Models & Pipelines =====
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)
from langchain.llms import (
    HuggingFaceHub,
    HuggingFacePipeline
)

# ===== RAG Evaluation (RAGAS) =====
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextRecall,
    ContextPrecision,
    AnswerCorrectness
)
from datasets import Dataset


# In[7]:


# because there was an error in this import, here it is seperatly
import torch
print(torch.__version__)


# In[8]:


from packaging import version

# Check torch version (modern alternative)
def is_torch_greater_or_equal_than_1_13():
    return version.parse(torch.__version__) >= version.parse("1.13.0")

print(f"Torch version: {torch.__version__}")
print(f"Is >=1.13.0: {is_torch_greater_or_equal_than_1_13()}")


# ## Authentication
# here we authenticate our LLM with hugging face

# In[9]:


# # Load environment variables from .env
# load_dotenv()

# # Retrieve the token
# hf_token = os.getenv("HUGGING_FACE_TOKEN")

# print("Token loaded:", hf_token is not None)

# # Log in to Hugging Face Hub
login(token="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")


# In[10]:


# config = dotenv_values(".env")
# login(token=config["HUGGING_FACE_TOKEN"])


# ## Constants
# here we define constants that we will fine tune

# In[29]:


# Constants
DEFAULT_CHUNK_SIZE = 1000              # Max size of each text chunk
DEFAULT_CHUNK_OVERLAP = 200            # Overlap between chunks
DEFAULT_SEARCH_K = 3                   # Top-k results to retrieve
DEFAULT_SEARCH_TYPE = "hybrid"         # Choose from: 'semantic', 'keyword', or 'hybrid'
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model for vector search
#DEFAULT_LLM_MODEL = "meta-llama/Llama-3.2-1B"   #worked!
# DEFAULT_LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # worked!

# DEFAULT_LLM_MODEL = "microsoft/phi-2"   #worked
# DEFAULT_LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2" #out of memory error
# DEFAULT_LLM_MODEL = "google/gemma-1.1-2b-it" #out of memory error
# DEFAULT_LLM_MODEL = "tiiuae/falcon-7b-instruct" #out of memory error
# LLM for generating answers
# DEFAULT_LLM_MODEL = "deepseek-ai/DeepSeek-V3-0324"                         # LLM for generating answers
# DEFAULT_LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_LLM_MODEL = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
# DEFAULT_DOCUMENT_DIR = "/data/corpus.zip"
DEFAULT_DOCUMENT_DIR = "/kaggle/input/daa-lectures-for-a4/cmu-lecs"
# DEFAULT_DOCUMENT_DIR = "/content/"  # Changed to the directory path

DEFAULT_LLM_MODEL


# # Document Processing Functions

# In[12]:


def load_documents(directory: str, glob_pattern: str = "**/*.pdf") -> List[Document]:
    """
    Loads all PDF files from a given directory.

    Args:
        directory: path to folder with PDF files
        glob_pattern: pattern to match files (default = all PDFs)

    Returns:
        List of LangChain Document objects
    """
    loader = DirectoryLoader(directory, glob=glob_pattern, loader_cls=PyPDFLoader)
    return loader.load()


# In[13]:


def chunk_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    separators: List[str] = None
) -> List[Document]:
    """
    Splits documents into chunks for better retrieval.

    Args:
        documents: list of LangChain documents
        chunk_size: size of each chunk
        chunk_overlap: how much content overlaps between chunks
        separators: optional list of separators for better splitting

    Returns:
        List of chunked Document objects
    """
    if separators is None:
        # Default separators: prioritize splitting on paragraphs, then sentences, then words
        separators = ["\n\n", "\n", ". ", " ", ""]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )
    return text_splitter.split_documents(documents)


# In[14]:


def create_vector_store(
    chunks: List[Document],
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    save_path: str = None
) -> FAISS:
    """
    Creates a FAISS vector index from document chunks using specified embedding model.

    Args:
        chunks: list of Document chunks
        embedding_model: HuggingFace model used for embeddings
        save_path: optional path to save the index

    Returns:
        FAISS vector store
    """
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectordb = FAISS.from_documents(chunks, embeddings)

    if save_path:
        vectordb.save_local(save_path)

    return vectordb


# In[15]:


def create_bm25_index(chunks: List[Document]) -> BM25Okapi:
    """
    Builds a keyword-based index using BM25.

    Args:
        chunks: list of Document chunks

    Returns:
        BM25 index
    """
    texts = [chunk.page_content for chunk in chunks]                  # Get plain text
    tokenized_texts = [text.split() for text in texts]               # Tokenize by whitespace
    return BM25Okapi(tokenized_texts)


# # Search Functions

# In[16]:


def semantic_search(
    query: str,
    vectordb: FAISS,
    k: int = DEFAULT_SEARCH_K,
    score_threshold: float = None
) -> List[Tuple[Document, float]]:
    """
    Perform semantic search using vector similarity from FAISS.

    Args:
        query: Natural language question
        vectordb: Vector index (FAISS)
        k: Number of results to return
        score_threshold: Filter out low similarity scores (optional)

    Returns:
        List of (Document, similarity_score) tuples
    """
    results = vectordb.similarity_search_with_score(query, k=k)

    # Optional thresholding to remove irrelevant results
    if score_threshold is not None:
        results = [(doc, score) for doc, score in results if score >= score_threshold]

    return results


# In[17]:


def keyword_search(
    query: str,
    bm25_index: BM25Okapi,
    chunks: List[Document],
    k: int = DEFAULT_SEARCH_K,
    score_threshold: float = None
) -> List[Tuple[Document, float]]:
    """
    Perform lexical search using BM25.

    Args:
        query: Search query string
        bm25_index: Pre-built BM25 index
        chunks: List of document chunks for mapping back
        k: Top-k documents to retrieve
        score_threshold: Optional filtering threshold for BM25 scores

    Returns:
        List of (Document, BM25_score) tuples
    """
    tokenized_query = query.split()  # Basic whitespace tokenization
    scores = bm25_index.get_scores(tokenized_query)

    # Get indices of top-k documents
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    results = [(chunks[i], scores[i]) for i in top_k_indices]

    if score_threshold is not None:
        results = [(doc, score) for doc, score in results if score >= score_threshold]

    return results


# In[18]:


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
    Combine semantic and keyword search using weighted score fusion.

    Args:
        query: Natural language query
        vectordb: FAISS vector database
        bm25_index: BM25 keyword index
        chunks: Document chunks (used for mapping back)
        k: Top-k results to return
        semantic_weight: Weight for vector similarity
        keyword_weight: Weight for BM25 relevance

    Returns:
        List of (Document, combined_score) tuples
    """
    # Step 1: Run both types of searches with larger k (to capture wider context)
    semantic_results = semantic_search(query, vectordb, k * 2)
    semantic_scores = {doc.page_content: score for doc, score in semantic_results}

    keyword_results = keyword_search(query, bm25_index, chunks, k * 2)
    keyword_scores = {doc.page_content: score for doc, score in keyword_results}

    # Step 2: Normalize BM25 scores (they are not bounded, unlike cosine similarity)
    max_kw_score = max(keyword_scores.values()) if keyword_scores else 1

    # Step 3: Combine results
    all_docs = set(semantic_scores.keys()).union(set(keyword_scores.keys()))
    combined_scores = []

    for doc_content in all_docs:
        sem_score = semantic_scores.get(doc_content, 0)
        kw_score = keyword_scores.get(doc_content, 0)
        norm_kw_score = kw_score / max_kw_score if max_kw_score > 0 else 0

        # Weighted sum of both types of scores
        combined_score = (semantic_weight * sem_score) + (keyword_weight * norm_kw_score)
        combined_scores.append((doc_content, combined_score))

    # Step 4: Sort and return top-k
    combined_scores.sort(key=lambda x: x[1], reverse=True)
    top_scores = combined_scores[:k]

    # Step 5: Re-map back to full Document objects using content
    doc_lookup = {chunk.page_content: chunk for chunk in chunks}
    results = []

    for doc_content, score in top_scores:
        if doc_content in doc_lookup:
            results.append((doc_lookup[doc_content], score))

    return results


# # LLM functions

# In[19]:


def initialize_llm(
    model_name: str = DEFAULT_LLM_MODEL,
    device: str = "cuda",  # Use "cpu" if not using GPU
    max_new_tokens: int = 300
) -> Tuple[pipeline, any]:
    """
    Loads a language model pipeline for text generation.

    Args:
        model_name: HuggingFace model repo (must support causal LM)
        device: "cuda" for GPU or "cpu"
        max_new_tokens: Max tokens to generate per response

    Returns:
        Tuple (generator pipeline, tokenizer)
    """
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype="auto",            # Uses GPU acceleration if available
            trust_remote_code=True         # Allow custom model architectures
        )
        print("Original used")
    except ImportError:
        # Fallback without device_map
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            trust_remote_code=True
        ).to(device)
        print("Edited used")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Format the prompt as a dialogue (user + assistant style)
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}User: {{ message['content'] }}\n"
        "{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}\n"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}Assistant:{% endif %}"
    )

    # Create a text-generation pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,        # Only return generated part, not the full prompt
        max_new_tokens=max_new_tokens,
        do_sample=True                 # Use sampling (stochastic generation)
    )

    return generator, tokenizer


# In[20]:


def generate_response(
    prompt: str,
    generator: pipeline,
    width: int = 80  # For pretty-printing long outputs
) -> str:
    """
    Generates a response from the LLM using the prompt.

    Args:
        prompt: Full RAG-formatted prompt with question + context
        generator: HF pipeline object
        width: max characters per printed line (for wrapping)

    Returns:
        Answer string
    """
    messages = [{"role": "user", "content": prompt}]        # Wrap prompt in chat message format
    output = generator(
        messages,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
                            # Call LLM
    return textwrap.fill(output[0]["generated_text"], width=width)


# In[21]:


def format_rag_prompt(
    question: str,
    retrieved_docs: List[Document],
    instruction: str = None
) -> str:
    """
    Formats the final input prompt to send to the LLM.

    Args:
        question: The user's natural language question
        retrieved_docs: List of document chunks retrieved by search
        instruction: Optional prompt instructions (system message)

    Returns:
        Full prompt text string
    """
    # Default instructions to guide the LLM on how to use retrieved documents
    if instruction is None:
        instruction = """You are an AI assistant tasked with answering questions based on retrieved knowledge.
                    - Integrate the key points from all retrieved responses into a cohesive, well-structured answer.
                    - If the responses are contradictory, mention the different perspectives.
                    - If none of the retrieved responses contain relevant information, reply:
                    "I couldn't find a good response to your query in the database."
                    """

    # Truncate each document to 1000 characters if long
    retrieved_info = "\n\n".join(
        f"{i+1}️⃣ {doc.page_content[:1000]}..." if len(doc.page_content) > 1000
        else f"{i+1}️⃣ {doc.page_content}"
        for i, doc in enumerate(retrieved_docs)
    )

    # Final structured prompt
    return f"""
        {instruction}

        ### Retrieved Information:
        {retrieved_info}

        ### Question:
        {question}
    """


# # RAG Evaluator

# In[22]:


# WRAPPER TO USE LANGCHAIN LLM IN RAGAS

class HuggingFaceLLMWrapper:
    """
    Wrapper for using HuggingFacePipeline with RAGAS evaluation.
    Includes a dummy 'set_run_config' to avoid errors.
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __call__(self, prompt: str) -> str:
        return self.pipeline(prompt)

    def set_run_config(self, run_config):
        """Dummy method to avoid errors with ragas."""
        pass  # Do nothing, as TextGenerationPipeline doesn't have this method


# In[23]:


# WRAPPER TO USE LANGCHAIN LLM IN RAGAS

class HuggingFaceLLM:
    """
    Simple wrapper for using HuggingFaceHub with RAGAS evaluation.
    """
    def __init__(self, model_name: str):
        self.model = HuggingFaceHub(repo_id=model_name)

    def __call__(self, prompt: str) -> str:
        return self.model(prompt)


# In[24]:


# EVALUATOR CLASS

class RAGEvaluator:
    """
    Performs automatic evaluation of RAG responses using Ragas metrics.
    Also supports result visualization and optimization insights.
    """
    def __init__(self, pipeline, llm, embeddings):
        self.pipeline = pipeline         # RAGPipeline object
        self.embeddings = embeddings     # HuggingFaceEmbeddings instance

        # Use passed LLM, or initialize default
        if isinstance(llm, str):
            self.llm = HuggingFaceHub(repo_id=llm)
        else:
            self.llm = HuggingFaceLLMWrapper(llm) if llm else HuggingFaceLLMWrapper(HuggingFaceHub(repo_id=DEFAULT_LLM_MODEL))

        # Internal result tracking
        self.results = []

    def evaluate_ragas(self, questions: list, gold_answers: list = None):
        """
        Run Ragas evaluation across all questions.

        Args:
            questions: List of input questions
            gold_answers: Reference answers (optional)

        Returns:
            DataFrame of results
        """
        all_rows = []

        for question, gold_answer in zip(questions, gold_answers or [None]*len(questions)):
            # Run full RAG query
            answer = self.pipeline.query(question)

            # Get context used in the answer
            contexts = [doc.page_content for doc in self.pipeline.get_last_retrieved_docs()]

            # Prepare a single sample for evaluation
            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts]
            }
            if gold_answer:
                data["ground_truth"] = [gold_answer]

            dataset = Dataset.from_dict(data)

            # Select metrics to compute
            metrics = [Faithfulness(), AnswerRelevancy(), ContextRecall(), ContextPrecision()]
            if gold_answer:
                metrics.append(AnswerCorrectness())

            # Run the evaluation
            result = evaluate(dataset, metrics=metrics, llm=self.llm, embeddings=self.embeddings)

            # Convert to DataFrame and store
            row = result.to_pandas()
            row["question"] = question
            row["retrieved_docs"] = len(contexts)
            all_rows.append(row)

        # Combine all rows into one DataFrame
        self.results = pd.concat(all_rows, ignore_index=True)
        return self.results

    def visualize_metrics(self):
        """
        Visualize average metric scores and context retrieval stats.
        """
        if self.results is None or self.results.empty:
            raise ValueError("No evaluation results found. Run evaluate_ragas() first.")

        # Plot main metrics
        metrics = ['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision']
        if 'answer_correctness' in self.results.columns:
            metrics.append('answer_correctness')

        plt.figure(figsize=(10, 5))
        self.results[metrics].mean().plot(kind='bar', color='lightblue')
        plt.title("🔍 Average RAG Evaluation Metrics")
        plt.ylabel("Score (0 to 1)")
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.show()

        # Plot document retrieval counts
        plt.figure(figsize=(8, 4))
        self.results['retrieved_docs'].value_counts().sort_index().plot(kind='bar', color='lightgreen')
        plt.title("📄 Number of Context Chunks Retrieved Per Query")
        plt.xlabel("Number of Chunks")
        plt.ylabel("Frequency")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.show()

    def get_optimization_insights(self):
        """
        Analyze weak metrics and recommend strategies to improve RAG performance.
        """
        if self.results is None or self.results.empty:
            return "No evaluation results available."

        insights = []
        df = self.results

        # Faithfulness issues (hallucination)
        if df['faithfulness'].mean() < 0.7:
            insights.append("⚠️ Faithfulness is low — possible hallucinations.")
            insights.append("🔧 Try increasing chunk overlap or improving retrieval relevance.")

        # Context recall issues (missing info)
        if df['context_recall'].mean() < 0.6:
            insights.append("⚠️ Low context recall — relevant info may be missed.")
            insights.append("🔧 Consider using hybrid retrieval or adjusting chunk size.")

        # Precision issues (irrelevant info)
        if df['context_precision'].mean() < 0.6:
            insights.append("⚠️ Low context precision — too much irrelevant context.")
            insights.append("🔧 Use better embeddings or rerank retrieved chunks.")

        # Relevance issues (answer not matching question)
        if df['answer_relevancy'].mean() < 0.7:
            insights.append("⚠️ Low answer relevancy — answers not matching question.")
            insights.append("🔧 Refine your prompts or improve chunk matching.")

        # Optional: Correctness based on gold answers
        if 'answer_correctness' in df.columns and df['answer_correctness'].mean() < 0.7:
            insights.append("⚠️ Low correctness — answers differ from references.")
            insights.append("🔧 Try different LLMs or use post-editing.")

        return "\n".join(insights)


# # RAG Pipeline

# In[25]:


from sentence_transformers import SentenceTransformer, util
import numpy as np

class SimpleEvaluator:
    """
    Evaluates the quality of generated answers by comparing the original question
    with a regenerated question from the generated answer.
    """
    def __init__(self, llm, tokenizer, embedding_model: str = DEFAULT_EMBEDDING_MODEL):
        self.llm = llm
        self.tokenizer = tokenizer
        self.embedding_model = SentenceTransformer(embedding_model)

    def generate_question(self, answer: str, max_chars: int = 200) -> str:
        """
        Generates a single-sentence question from a given answer.
    
        Args:
            answer: The answer text to reverse-generate a question for.
            max_chars: Maximum character length of the output question.
    
        Returns:
            A short, single question string.
        """
        prompt = (
            "You are a question generation AI.\n"
            "Generate exactly one concise, clear question (no explanations) that can be answered using the following passage.\n\n"
            f"PASSAGE:\n{answer}\n\n"
            "QUESTION (one line only):"
        )
    
        messages = [{"role": "user", "content": prompt}]
        output = self.llm(
            messages,
            max_new_tokens=64,          # prevent long rambly generations
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
    
        # Get the first sentence and truncate if necessary
        full_output = output[0]["generated_text"].strip()
        single_line = full_output.split("\n")[0].strip()  # grab first line
        trimmed = single_line[:max_chars].strip()
    
        # Ensure it ends with a question mark
        if not trimmed.endswith("?"):
            trimmed += "?"
    
        return trimmed

    def compute_faithfulness(self, generated_answer: str, retrieved_docs: list) -> float:
        """
        Computes faithfulness score between answer and retrieved docs.
        """
        combined_context = " ".join([doc.page_content for doc in retrieved_docs])
        emb_ans = self.embedding_model.encode(generated_answer, convert_to_tensor=True)
        emb_ctx = self.embedding_model.encode(combined_context, convert_to_tensor=True)
        return float(util.pytorch_cos_sim(emb_ans, emb_ctx).item())

    def compute_similarity(self, q1: str, q2: str) -> float:
        emb1 = self.embedding_model.encode(q1, convert_to_tensor=True)
        emb2 = self.embedding_model.encode(q2, convert_to_tensor=True)
        return float(util.pytorch_cos_sim(emb1, emb2).item())

    def evaluate(self, question: str, generated_answer: str, retrieved_docs: list) -> dict:
        regenerated_q = self.generate_question(generated_answer)
        q_similarity = self.compute_similarity(question, regenerated_q)
        faith_score = self.compute_faithfulness(generated_answer, retrieved_docs)
        return {
            "original_question": question,
            "regenerated_question": regenerated_q,
            "question_similarity": round(q_similarity, 4),
            "faithfulness_score": round(faith_score, 4)
        }


# In[26]:


# MAIN RAG PIPELINE

class RAGPipeline:
    """
    This is the central class that handles:
    - Loading and chunking documents
    - Initializing vector and keyword search
    - Running queries
    - Generating responses from the LLM
    - Running full experiment sweeps
    """
    def __init__(
        self,
        document_dir: str,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        llm_model: str = DEFAULT_LLM_MODEL,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        device: str = "cuda"
    ):
        self.document_dir = document_dir
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.device = device

        # To store runtime state
        self.documents = None
        self.chunks = None
        self.vectordb = None
        self.bm25_index = None
        self.llm = None
        self.tokenizer = None
        self.last_retrieved_docs = None  # For evaluation traceability


    # ==========================
    # LOAD & CHUNK DOCUMENTS
    # ==========================
    def load_and_process_documents(self):
        """
        Loads PDF documents and splits them into overlapping chunks.
        """
        print("📄 Loading documents...")
        self.documents = load_documents(self.document_dir)
        print(f"✅ Loaded {len(self.documents)} document pages.")

        print("🪓 Chunking documents...")
        self.chunks = chunk_documents(
            self.documents,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        print(f"✅ Created {len(self.chunks)} chunks.")

        # Add unique IDs to chunks for tracking
        for i, chunk in enumerate(self.chunks):
            chunk.metadata["chunk_id"] = i


    # ==========================
    # INITIALIZE RETRIEVAL
    # ==========================
    def initialize_retrieval(self):
        """
        Builds vector store and keyword index for retrieval.
        """
        if not self.chunks:
            raise ValueError("❌ No chunks found. Run load_and_process_documents() first.")

        print("📦 Creating vector store...")
        self.vectordb = create_vector_store(self.chunks, self.embedding_model)

        print("🔎 Creating BM25 index...")
        self.bm25_index = create_bm25_index(self.chunks)


    # ==========================
    # INITIALIZE LLM
    # ==========================
    def initialize_llm(self):
        """
        Loads the chosen LLM and tokenizer from HuggingFace.
        """
        print("🤖 Loading LLM...")
        self.llm, self.tokenizer = initialize_llm(self.llm_model, self.device)


    # ==========================
    # GET LAST RETRIEVED CHUNKS
    # ==========================
    def get_last_retrieved_docs(self):
        """
        Returns the last set of retrieved document chunks (used in evaluation).
        """
        if self.last_retrieved_docs is None:
            raise ValueError("❌ No retrievals done yet.")
        return self.last_retrieved_docs


    # ==========================
    # MAIN QUERY FUNCTION
    # ==========================
    def query(
        self,
        question: str,
        search_type: str = DEFAULT_SEARCH_TYPE,  # semantic / keyword / hybrid
        k: int = DEFAULT_SEARCH_K,
        semantic_weight: float = 0.5,
        keyword_weight: float = 0.5,
        custom_instruction: str = None
    ) -> str:
        """
        Executes a full query through the pipeline:
        - Retrieves chunks
        - Formats prompt
        - Calls LLM
        - Returns answer
        """
        if not self.vectordb or not self.bm25_index:
            raise ValueError("❌ Retrieval systems not ready. Run initialize_retrieval().")
        if not self.llm:
            raise ValueError("❌ LLM not initialized. Run initialize_llm().")

        # Step 1: Retrieve relevant documents
        if search_type == "semantic":
            results = semantic_search(question, self.vectordb, k)
        elif search_type == "keyword":
            results = keyword_search(question, self.bm25_index, self.chunks, k)
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
        else:
            raise ValueError(f"❌ Unknown search type: {search_type}")

        retrieved_docs = [doc for doc, _ in results]
        self.last_retrieved_docs = retrieved_docs

        # Step 2: Format prompt
        prompt = format_rag_prompt(question, retrieved_docs, custom_instruction)

        # Step 3: Generate LLM response
        return generate_response(prompt, self.llm)


    # ==========================
    # EXPERIMENTATION FUNCTION
    # ==========================
    def experiment(
        self,
        questions: List[str],
        gold_answers: List[str],
        chunk_sizes: List[int],
        k_values: List[int],
        search_types: List[str],
        chunk_overlaps: List[int] = [0, 100, 200]
    ) -> Dict:
        """
        Run multiple experiment configurations (varying chunk size, k, search type).

        Args:
            questions: list of input queries
            gold_answers: reference answers (used in evaluation)
            chunk_sizes: different chunk sizes to test
            k_values: number of results to retrieve
            search_types: list of retrieval modes
            chunk_overlaps: amount of content overlap between chunks

        Returns:
            A dict of results by config
        """
        results = {}

        for chunk_size in chunk_sizes:
            for chunk_overlap in chunk_overlaps:
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap

                print(f"\n⚙️  Testing: chunk={chunk_size}, overlap={chunk_overlap}")
                self.load_and_process_documents()
                self.initialize_retrieval()

                evaluator = RAGEvaluator(self, self.llm, HuggingFaceEmbeddings(model_name=self.embedding_model))

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

                                # Run evaluation (with gold answer)
                                eval_result = evaluator.evaluate_ragas([question], [gold_answers[0]])
                                print(eval_result[['faithfulness', 'answer_relevancy']])

                                # Visualization + optimization
                                evaluator.visualize_metrics()
                                print("\n🧠 Optimization Suggestions:")
                                print(evaluator.get_optimization_insights())

                            except Exception as e:
                                results[config_name][question] = f"❌ Error: {str(e)}"

        return results


    def grid_search(
        self,
        questions: List[str],
        gold_answers: List[str],
        chunk_sizes: List[int],
        k_values: List[int],
        search_types: List[str],
        chunk_overlaps: List[int] = [0, 100, 200],
        semantic_weight: float = 0.5,
        keyword_weight: float = 0.5,
        output_csv_path: str = "rag_grid_search_results.csv"
    ) -> Dict:
        """
        Run grid search over multiple config combinations and log results to CSV.

        Returns:
            Dictionary of results.
        """
        results = []
        header_written = False

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)

        with open(output_csv_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            for chunk_size in chunk_sizes:
                for chunk_overlap in chunk_overlaps:
                    if chunk_overlap >= chunk_size:
                        continue
                    self.chunk_size = chunk_size
                    self.chunk_overlap = chunk_overlap

                    self.load_and_process_documents()
                    self.initialize_retrieval()

                    evaluator = SimpleEvaluator(self.llm, self.tokenizer)

                    for search_type in search_types:
                        for k in k_values:
                            config_name = f"chunk{chunk_size}_overlap{chunk_overlap}_{search_type}_k{k}"
                            print(f"\n🔬 Running Config: {config_name}")

                            for i, question in enumerate(questions):
                                try:
                                    start_time = time.time()

                                    # Step 1: Query
                                    retrieval_start = time.time()
                                    answer = self.query(
                                        question=question,
                                        search_type=search_type,
                                        k=k,
                                        semantic_weight=semantic_weight,
                                        keyword_weight=keyword_weight
                                    )
                                    retrieval_end = time.time()

                                    # Step 2: Evaluation
                                    eval_df = evaluator.evaluate([question], [answer], self.last_retrieved_docs)
                                    # eval_row = eval_df.iloc[0].to_dict()

                                    # Step 3: Timing
                                    end_time = time.time()
                                    total_time = end_time - start_time
                                    retrieval_time = retrieval_end - retrieval_start
                                    generation_time = total_time - retrieval_time

                                    print(eval_df)

                                    # # Visualization + optimization
                                    # evaluator.visualize_metrics()
                                    # print("\n🧠 Optimization Suggestions:")
                                    # print(evaluator.get_optimization_insights())

                                    # Step 4: Build result row
                                    row = {
                                        "time run":  print_date_time(),
                                        "config": config_name,
                                        "question": question,
                                        "answer": answer,
                                        "retrieved_documents": self.last_retrieved_docs,
                                        "chunk_size": chunk_size,
                                        "chunk_overlap": chunk_overlap,
                                        "search_type": search_type,
                                        "top_k": k,
                                        "semantic_weight": semantic_weight,
                                        "keyword_weight": keyword_weight,
                                        "LLM": self.llm_model,
                                        "embedding_model": self.embedding_model,
                                        "retrieval_time": round(retrieval_time, 4),
                                        "generation_time": round(generation_time, 4),
                                        "total_time": round(total_time, 4),
                                        **eval_df
                                    }

                                    # Write CSV header if needed
                                    if not header_written:
                                        writer.writerow(row.keys())
                                        header_written = True

                                    writer.writerow(row.values())
                                    results.append(row)

                                except Exception as e:
                                    print(f"⚠️ Error in config {config_name} for question '{question}': {e}")
                                    writer.writerow([config_name, question, f"Error: {str(e)}"])

        print(f"\n✅ All results logged to: {output_csv_path}")
        return results


# # Running
# here we now run the code

# In[30]:


# STEP 1: CREATE EMBEDDINGS & LLM WRAPPER

# Use the same model you defined as DEFAULT_EMBEDDING_MODEL
embeddings = HuggingFaceEmbeddings(model_name=DEFAULT_EMBEDDING_MODEL)

# Initialize LLM pipeline
generator, tokenizer = initialize_llm(
    model_name=DEFAULT_LLM_MODEL,     # <-- Use the same LLM constant from above
    device="cuda",                    # "cuda" or "cpu"
    max_new_tokens=500                # <-- You can increase this if responses are too short
)

# Wrap your generator for LangChain compatibility
local_llm = HuggingFacePipeline(pipeline=generator)


# In[31]:


# STEP 2: INITIALIZE RAG PIPELINE

rag = RAGPipeline(
    document_dir=DEFAULT_DOCUMENT_DIR,       # <-- uses constant
    embedding_model=DEFAULT_EMBEDDING_MODEL,
    llm_model=DEFAULT_LLM_MODEL,
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    device="cuda"
)
rag


# In[32]:


# STEP 3: LOAD DOCUMENTS & PREPARE SYSTEM

rag.load_and_process_documents()
rag.initialize_retrieval()
rag.initialize_llm()  # This will use Llama model defined above


# In[34]:


# STEP 4: ASK A SINGLE QUESTION

questions = [
    "What is Dynamic Programming?"
    # "Explain the matrix method in hashing",
    # "What are the key concepts in amortized analysis?"
]

# Run query with hybrid search and show result
answer = rag.query(
    questions[0],
    search_type=DEFAULT_SEARCH_TYPE,
    k=DEFAULT_SEARCH_K
)
print(f"\n🧠 Answer:\n{answer}")


# In[39]:


# STEP 5: PROVIDE GROUND TRUTH FOR EVALUATION (OPTIONAL)

gold_answers = [
    "Dynamic Programming is a technique for solving problems by breaking them into overlapping subproblems, storing intermediate results, and combining them to solve the larger problem efficiently."
    # "Dynamic Programming is a powerful technique that can be used to solve many combinatorial problems in polynomial time for which a naive approach would take exponential time. Dynamic Programming is a general approach to solving problems, much like “divide-and-conquer”, except that the subproblems will overlap."
]


# In[ ]:


# STEP 6: RUN AUTOMATIC EVALUATION

evaluator = RAGEvaluator(rag, llm=local_llm, embeddings=embeddings)
results_df = evaluator.evaluate_ragas(questions, gold_answers)
print("\n📊 Evaluation Results:\n", results_df)


# In[ ]:


# Visualize performance
evaluator.visualize_metrics()


# In[ ]:


# Show optimization tips
print("\n🛠️ Suggestions to Improve RAG System:")
print(evaluator.get_optimization_insights())


# In[40]:


experiment_results = rag.grid_search(
    questions=questions,
    gold_answers=gold_answers,
    chunk_sizes=[100, 250, 500, 800, 1000],
    k_values=[3, 4, 5],
    search_types=["semantic", "hybrid", "keyword"],
    chunk_overlaps=[100, 200],
    # output_csv_path="/content/rag_grid_log.csv"
    output_csv_path="/kaggle/working/rag_grid_log.csv"
)

