# Assignment 4 - RAG-Based Question Answering System

## Overview

This assignment aimed to build a **Retrieval-Augmented Generation (RAG)** based Question Answering system that can answer questions using information retrieved from a set of documents. Instead of relying solely on the internal knowledge of language models, our system first retrieves relevant document chunks using TF-IDF, and then generates answers based on these retrieved chunks using a pre-trained language model.

## Objective

- Implement a complete RAG pipeline using TF-IDF for retrieval and HuggingFace language models for answer generation.
- Evaluate how well different language models perform in this setup.
- Use quantitative metrics (BLEU and ROUGE) to assess the quality of generated answers.
- Analyze how the number of retrieved documents affects answer quality.

## What We Built

- **Retriever**: TF-IDF-based vectorizer to fetch the top-k relevant document chunks for each question.
- **Generator**: We tested two HuggingFace models — **LLaMA-2-7B** and **Microsoft Phi-2** — to generate answers based on the retrieved documents.
- **Evaluator**: Computed BLEU and ROUGE scores by comparing the generated answers to reference answers from a synthetic SQuAD-like dataset.

## Experiments and Evaluation

We tested:
- Different numbers of retrieved documents (top-1, top-3, top-5) to observe the impact of context size.
- Performance of both **LLaMA-2-7B** and **Phi-2** models on the same dataset.

**Evaluation Metrics**:
- **BLEU**: Measures word overlap with the reference answer.
- **ROUGE**: Measures n-gram recall and overlap for fluency and content.

## Results & Findings

- **Phi-2 consistently outperformed LLaMA-2** across BLEU and ROUGE metrics.
- Retrieving **top-3 documents** gave the best balance between relevance and noise.
- The quality of retrieval was crucial — irrelevant documents led to hallucinated or wrong answers.
- Overall, RAG with Phi-2 and a top-3 retrieval strategy proved most effective.

## Conclusion

This assignment demonstrated how retrieval can significantly enhance the factual accuracy of generative QA systems. Even lightweight models like **Microsoft's Phi-2**, when combined with simple retrieval techniques like TF-IDF, can generate high-quality answers. The pipeline is modular and allows easy extension to other models or retrieval strategies in future work.

## Running the Code

```bash
pip install -r requirements.txt
python rag_qa_system.py
```
This will:
- Load the dataset
- Retrieve relevant documents for each question
- Generate answers using both models
- Print BLEU and ROUGE evaluation scores