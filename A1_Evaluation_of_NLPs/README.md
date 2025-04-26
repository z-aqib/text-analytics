# Assignment 1
need to pass an input text to three different models. and analyse their summarization, translation, question answering and time effeicency. 

## 3B Params
- mistral 3B is not supported by huggingface
- i ran qwen, it was fast (25s) and it was really really good. french-to-english was 12s and was completely same as input text. the keyword extraction however wasnt good as it just extracted sentences from text, not words. but overall it was amazing, it was quick, effieicent, and concise and complete

## 7B Params
- falcon 7B took way too long (8mins) and was only running the 1st prompt (summary) and was just repeatedly copy pasting the same 5 words again and again. french-to-english translation was good (7mins), however again it was just copy pasting again again.
- so i switched to openthinker, it was long again (9mins) but good, however i had ran it without the translation prompt so im running it again with transaltion prompt. it was again repeating the entire prompt until its tokens were full
- so in second running openthinker, it gave a quick answer (2mins)

## 13B Params
- microsoft phi worked good but it skipped things itself

---

# Assignment 01 - LLM Response Assessment

**Course:** Introduction to Text Analytics  
**Name:** Zuha Aqib  
**ID:** 26106  

## Overview
This assignment involved evaluating three LLMs across multiple tasks based on specific criteria:

- **3B Model:** Qwen/Qwen2.5-3B-Instruct
- **7B Model:** open-thoughts/OpenThinker-7B
- **14B Model:** microsoft/phi-4

Tasks included Summarization, Question Answering, Keyword Extraction, and Translation. Each was scored from 1-5 on criteria like Conciseness, Clarity, Accuracy, Completeness, Fidelity, Fluency, and Consistency.

## Key Findings
- **Best Overall Model:** **Qwen (3B)** — Fastest, most accurate, and completed all tasks properly.
- **Keyword Extraction:** **OpenThinker (7B)** performed best.
- **Time Efficiency:** Qwen was the fastest, completing tasks significantly quicker than the others.
- **Suitability for IBA:** Qwen is recommended for its speed, accuracy, and usefulness for students, teachers, and administrative staff needing summarization and question-answering tasks.

## Evaluation Criteria
- **Summarization:** Conciseness, Clarity
- **Question Answering:** Accuracy, Completeness
- **Keyword Extraction:** Completeness, Categorization
- **Translation:** Fidelity, Fluency, Consistency
