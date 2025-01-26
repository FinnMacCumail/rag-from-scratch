# Rank Fusion and Query Generation with LangChain
![alt text](rank-fusion.png)
This document explains the provided code snippet, focusing on the rank fusion process using Reciprocal Rank Fusion (RRF) and its integration with LangChain for generating and retrieving context-aware search results.

## Code Overview

### Query Generation

The query generation section uses a predefined ChatPromptTemplate to dynamically create search queries related to a given question.

### Template Definition:
```python
template = """You are a helpful assistant that generates multiple search queries based on a single input query.
Generate multiple search queries related to: {question}
Output (4 queries):"""
```

This template instructs the AI model to generate four distinct search queries based on an input question.

### Query Chain:
```python
prompt_rag_fusion = ChatPromptTemplate.from_template(template)
question = "What is task decomposition for LLM agents?"
```
# Chain to generate queries
```python
generate_queries = (
    prompt_rag_fusion
    | llm
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)
```
### This chain:

1. Applies the template using the prompt_rag_fusion object.

2. Processes the template with the language model (llm) to generate output.

3. Parses the output into a string using StrOutputParser().

4. Splits the parsed output into a list of individual queries.

## Reciprocal Rank Fusion (RRF)

The RRF algorithm combines ranked lists of documents into a single reranked list by assigning scores based on their ranks in the input lists.

### Function:
```python

def reciprocal_rank_fusion(results: list[list], k=60):
    # Initialize fused scores for unique documents
    fused_scores = {}

    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)  # Serialize the document
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)  # RRF formula

    # Sort documents by their fused scores
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    return reranked_results
```
### Key Steps:

1. Initialization:

A dictionary fused_scores tracks cumulative RRF scores for documents.

2. Iterate through input results:

- results is a list of ranked lists, where each list contains documents.

- Each document is serialized using dumps() to ensure consistent handling as dictionary keys.

3. Scoring with RRF Formula:

- For each document, its score is incremented by 1 / (rank + k), where:

- rank is its position in the current list (0-indexed).

- k is a tunable parameter controlling the impact of rank.

4. Sort and Return:

- Documents are sorted in descending order of their cumulative scores.

- Each document is deserialized using loads() before being returned with its score.

## Retrieval and Reranking Chain

The generated queries are passed to a retriever for fetching relevant documents, followed by rank fusion.

### Chain Construction:
```python
retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
```

1. generate_queries: Produces multiple search queries.

2. retriever.map(): Maps each query to its retrieved documents.

3. reciprocal_rank_fusion: Combines the retrieved results using RRF.

### Execution:
```python
docs = retrieval_chain_rag_fusion.invoke({"question": question})
len(docs)
```

- invoke********: Executes the chain to retrieve and rank documents.

- len(docs): Returns the number of reranked documents.

## RAG (Retrieval-Augmented Generation)

The RAG process uses retrieved documents to generate context-aware answers.

### Contextual Prompt:
```python
template = """Answer the following question based on this context:

{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
```
This template combines retrieved context with the input question to generate an answer.

### Final RAG Chain:
```python
final_rag_chain = (
    {"context": retrieval_chain_rag_fusion,
     "question": itemgetter("question")}
    | prompt
    | llm
    | StrOutputParser()
)

final_rag_chain.invoke({"question":question})
```
1. Input Mapping:

- context is populated with reranked results from retrieval_chain_rag_fusion.

- question is extracted from the input.

2. Prompting:

- A contextualized prompt is created using the prompt object.

3. Response Generation:

- The language model generates a response based on the prompt.

4. Output Parsing:

- The response is parsed into a structured output using StrOutputParser().

## Summary

This code demonstrates a robust pipeline for retrieval-augmented generation (RAG):

1. Query Generation: Dynamically create multiple search queries.

2. Retrieval: Fetch documents for each query using a retriever.

3. Rank Fusion: Combine and rerank retrieved results with Reciprocal Rank Fusion.

4. Answer Generation: Use the reranked results as context to answer the input question.

By integrating rank fusion into the RAG pipeline, the system effectively balances relevance across multiple retrieval strategies, ensuring a more comprehensive and accurate response.