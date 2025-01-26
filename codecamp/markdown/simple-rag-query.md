# Explanation of RAG Pipeline Code with LangChain and Ollama

This code demonstrates a **retrieval-augmented generation (RAG)** pipeline using LangChain and Ollama. The pipeline retrieves and embeds documents from a web page, stores them in a vector database, and generates an answer to a query.

---

## Code Breakdown

### 1. Import Statements
The following modules are used:
- **Document Loading and Parsing**:
  - `WebBaseLoader`: Scrapes web content.
  - `bs4`: Parses HTML using BeautifulSoup.
- **Text Splitting**:
  - `RecursiveCharacterTextSplitter`: Splits documents into chunks.
- **Embeddings**:
  - `OllamaEmbeddings`: Generates dense vector embeddings.
- **Vectorstore**:
  - `Chroma`: Stores document embeddings.
- **RAG Components**:
  - `hub`: Fetches pre-defined prompt templates.
  - `ChatOllama`: Handles language model inference.
  - `StrOutputParser`: Formats model outputs.
  - `RunnablePassthrough`: Chains steps together.

---

## Indexing Phase

### 1. Load Documents
The `WebBaseLoader` fetches web content from the specified URL:

```python
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()
```

### 2. Split Documents
The `RecursiveCharacterTextSplitter` splits documents into smaller chunks:

```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
```

### 3. Create Vector Embeddings

Embeddings are created using OllamaEmbeddings:
```python
embedding_function = OllamaEmbeddings(model='mxbai-embed-large')
```
The Chroma vectorstore stores these embeddings:
```python
persistent_directory = os.path.join(current_dir, "db", "chroma_db")
vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=embedding_function, persist_directory=persistent_directory)
vectorstore.persist()
```
### 4. Create a Retriever
The vectorstore is converted into a retriever for similarity-based search:
```python
retriever = vectorstore.as_retriever()
```

## Retrieval and Generation Phase
### 1. Define Prompt
A pre-defined prompt template is fetched:
```python
prompt = hub.pull("rlm/rag-prompt")
```
## 2. Language Model
An Ollama model is initialized for generating responses:
## 3. Define the RAG Chain
```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```
Steps:
1. Retrieve relevant documents using retriever.
2. Format documents with format_docs.
3. Combine context and query using the prompt.
4. Generate an answer using ChatOllama.
5. Parse the output with StrOutputParser.
## 4. Query the Chain
```python
rag_chain.invoke("What is Task Decomposition?")
```

## Key Features
- **Key Features**:

    - Web-Based Document Retrieval:
        Fetches and filters specific content from web pages.

    - Chunk-Based Processing:
        Splits large documents into smaller, overlapping chunks for better embedding and retrieval.

    - Persistent Vectorstore:
        Saves embeddings locally for future reuse.

    - RAG Chain:
        Combines document retrieval and language generation for accurate, context-aware answers.

