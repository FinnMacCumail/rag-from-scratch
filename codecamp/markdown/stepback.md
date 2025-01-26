# Few-Shot Examples with Step-Back Question Generation and Contextual Answering
![alt text](step-back.png)
This document explains the provided code, which demonstrates the use of **few-shot prompting**, **step-back question generation**, and **contextual answering** using LangChain.

---

## Overview

### Key Features:
1. **Few-Shot Prompting:** Teaches the AI model to reframe questions into more generic, step-back versions using example pairs.
2. **Step-Back Question Generation:** Produces generalized forms of questions to retrieve broader context.
3. **Contextual Answering:** Combines context retrieved from both original and step-back questions to generate a comprehensive answer.

---

## Code Walkthrough

### Few-Shot Examples

#### Example Data
```python
examples = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "what can the members of The Police do?",
    },
    {
        "input": "Jan Sindel’s was born in what country?",
        "output": "what is Jan Sindel’s personal history?",
    },
]
```
These examples demonstrate how to reframe specific questions into generalized, step-back questions.
Example Prompt
```python
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
```
The example_prompt defines the format for each example:

- `"human": Represents the input question.
- `"ai": Represents the generalized, step-back version of the question.

### Few-Shot Prompt Construction
```python
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)
```
The FewShotChatMessagePromptTemplate integrates the examples into the prompt. The AI uses these examples as guidance for generating step-back questions.
### Main Prompt for Step-Back Question Generation
```python
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
        ),
        # Few shot examples
        few_shot_prompt,
        # New question
        ("user", "{question}"),
    ]
)
```
1. System Message: Provides instructions to the model.
2. Few-Shot Examples: Shows the AI how to perform the task using prior examples.
3. User Message: Supplies the input question to be paraphrased.

### Execution
```python
generate_queries_step_back = prompt | llm | StrOutputParser()
question = "What is task decomposition for LLM agents?"
generate_queries_step_back.invoke({"question": question})
```
1. The input question ("What is task decomposition for LLM agents?") is processed by the prompt and model (llm).
2. The output is a step-back version of the question.

### Response Generation with Context
#### Response Prompt Template
```python
response_prompt_template = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.

# {normal_context}
# {step_back_context}

# Original Question: {question}
# Answer:"""
response_prompt = ChatPromptTemplate.from_template(response_prompt_template)
```
This prompt incorporates:

1. normal_context: Context retrieved using the original question.
2. step_back_context: Context retrieved using the step-back question.
3. question: The original question.
4. Instruction: Guides the model to ensure relevance and comprehensiveness in its answer.

### Combined Chain for Retrieval and Answering
```python
chain = (
    {
        # Retrieve context using the normal question
        "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
        # Retrieve context using the step-back question
        "step_back_context": generate_queries_step_back | retriever,
        # Pass on the question
        "question": lambda x: x["question"],
    }
    | response_prompt
    | llm
    | StrOutputParser()
)
```
### Breakdown:

1. Context Retrieval:
    - normal_context: Directly retrieves context using the original question.
    - step_back_context: Uses the step-back question to retrieve additional context.
2. Response Prompting:
    - Combines both contexts and the original question to create the response prompt.
3. LLM and Parsing:
    - The language model generates the answer.
    - StrOutputParser() parses the model's output into a structured response.

### Execution:
```python
chain.invoke({"question": question})
```
- Combines all steps into a single chain to generate the final response.

## Summary of Functionality
### Input:

### Original Question:
What is task decomposition for LLM agents?
Process:

1. Step-Back Question Generation:
    -Converts the original question into a more generic form, e.g., What are the broad principles behind task breakdown in AI systems?
2. Context Retrieval:
    -Retrieves context for both the original and step-back questions using the retriever.
3. Response Generation:
    - Uses the retrieved contexts to generate a comprehensive and contextually relevant answer.

### Output:

A well-structured answer that balances specificity and generality, leveraging insights from both the original and step-back contexts.

