# Code Explanation: Decomposition for LLM-Powered Autonomous Agent Systems
![alt text](save-recursively.png)
This document explains the provided code, focusing on the **decomposition** process where an input question is broken down into sub-questions and processed to generate a comprehensive answer using LangChain and ChatOllama.

---

## Overview

The code achieves the following:
1. Decomposes a high-level input question into sub-questions.
2. Iteratively retrieves context and generates answers for the sub-questions.
3. Formats the output as question-answer (Q&A) pairs.

---

## Key Components

### 1. **Decomposition Query Generation**

#### Template for Sub-Question Generation
```python
template = """You are a helpful assistant that generates multiple sub-questions related to an input question. 
The goal is to break down the input into a set of sub-problems / sub-questions that can be answered in isolation. 
Generate multiple search queries related to: {question} 
Output (3 queries):"""
prompt_decomposition = ChatPromptTemplate.from_template(template)
```

This template instructs the assistant to generate 3 sub-questions for a given input question.
## Query Chain
```python
llm = ChatOllama(model="llama3.1")

# Chain for sub-question generation
generate_queries_decomposition = (
    prompt_decomposition
    | llm
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)
```
- prompt_decomposition: Constructs the prompt for sub-question generation.
- llm: Uses the ChatOllama llama3.1 model to generate results.
- StrOutputParser(): Converts the LLM output into a string format.
- (lambda x: x.split("\n")): Splits the string into a list of sub-questions.

1. Execution
```python

question = "What are the main components of an LLM-powered autonomous agent system?"
questions = generate_queries_decomposition.invoke({"question": question})

    The input question is decomposed into a list of sub-questions stored in questions.
```
2. Answer Generation
### Prompt for Contextual Answering
```python

template = """Here is the question you need to answer:

--- 
{question} 
---

Here is any available background question + answer pairs:

--- 
{q_a_pairs} 
---

Here is additional context relevant to the question: 

--- 
{context} 
---

Use the above context and any background question + answer pairs to answer the question: 
{question}
"""
decomposition_prompt = ChatPromptTemplate.from_template(template)
```
This prompt combines:

1. The question to be answered.
2. Background Q&A pairs that provide historical context.
3. Additional context retrieved for the specific question.

### Function for Formatting Q&A Pairs
```python
def format_qa_pair(question, answer):
    """Format Q and A pair"""
    formatted_string = f"Question: {question}\nAnswer: {answer}\n\n"
    return formatted_string.strip()
```
This function structures the Q&A pairs for readability and consistency.
### Iterative Q&A Processing
```python
q_a_pairs = ""

for q in questions:
    rag_chain = (
        {"context": itemgetter("question") | retriever, 
         "question": itemgetter("question"),
         "q_a_pairs": itemgetter("q_a_pairs")}
        | decomposition_prompt
        | llm
        | StrOutputParser()
    )
    answer = rag_chain.invoke({"question": q, "q_a_pairs": q_a_pairs})
    q_a_pair = format_qa_pair(q, answer)
    q_a_pairs = q_a_pairs + "\n---\n" + q_a_pair
```
1. For Each Sub-Question (q):
    - Retrieves context using a retriever based on the current sub-question.
    - Combines the context, sub-question, and Q&A pairs into a structured prompt (decomposition_prompt).
    - Generates an answer using the llm.
    - Formats the sub-question and answer as a Q&A pair.
2. Cumulative Q&A Pairs: The generated Q&A pairs are appended to q_a_pairs, providing context for subsequent iterations.

## Summary of the Pipeline
### Input:

### High-Level Question:
"What are the main components of an LLM-powered autonomous agent system?"
Steps:

1. Decompose the Input Question:
    - Break the question into 3 sub-questions using generate_queries_decomposition.
2. Iterative Contextual Answering:
    - For each sub-question:
        - Retrieve relevant context.
        - Generate an answer.
        - Append the Q&A pair to q_a_pairs.
3. Final Output:
    - A complete set of Q&A pairs providing detailed insights into the input question.

### Key Concepts:

- Decomposition: Breaking down a complex question into manageable sub-questions.
- Contextual Retrieval: Using retrievers to fetch relevant information for each sub-question.
- Q&A Formatting: Ensuring the output is readable and logically organized.
