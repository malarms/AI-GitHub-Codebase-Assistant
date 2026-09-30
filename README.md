# AI GitHub Codebase Assistant

A RAG-based codebase question-answering system that ingests a GitHub repository, parses its source code, generates semantic embeddings, stores code chunks in Qdrant, retrieves relevant code using dense semantic search, and generates grounded answers with file and line citations.

## Overview

Understanding an unfamiliar codebase often requires manually searching through files, tracing functions, and following implementation details.

This project builds an AI-powered codebase assistant that allows developers to ask questions about a repository in natural language.

For example:

> How does the application make an HTTP request?

The system retrieves the most relevant code from the indexed repository and passes it to an LLM, which generates an answer grounded in the retrieved source code.

The generated response includes citations to the relevant files and line ranges.

---

## Architecture

```text
GitHub Repository
        │
        ▼
Repository Ingestion
        │
        ▼
Code Parsing
(AST / code-aware chunks)
        │
        ▼
BGE Embeddings
        │
        ▼
Qdrant Vector Database
        │
        ▼
Dense Semantic Retrieval
        │
        ▼
Retrieved Code Context
        │
        ▼
LLM
        │
        ▼
Grounded Answer
+ File / Line Citations
        │
        ▼
FastAPI
        │
        ▼
Docker
```

---

## Key Features

- GitHub repository ingestion
- Code-aware source parsing
- Python AST-based function and method extraction
- Multi-language file discovery and parsing support
- Semantic code embeddings using BGE
- Persistent local Qdrant vector database
- Dense semantic retrieval
- Retrieval-Augmented Generation (RAG)
- File and line-level source citations
- FastAPI REST API
- Dockerized deployment
- Retrieval evaluation using Recall@5 and MRR

---

## Tech Stack

### AI / Retrieval

- Python
- Sentence Transformers
- BAAI/BGE-small-en-v1.5
- Qdrant
- Retrieval-Augmented Generation (RAG)

### Backend

- FastAPI
- Pydantic
- OpenAI-compatible LLM API

### Infrastructure

- Docker
- Git
- GitHub

---

## Project Structure

```text
github-codebase-rag/
│
├── app/
│   ├── api/
│   │   └── main.py
│   │
│   ├── evaluation/
│   │   └── run_evaluation.py
│   │
│   ├── generation/
│   │   ├── llm.py
│   │   ├── rag.py
│   │   └── run_rag.py
│   │
│   ├── ingestion/
│   │   ├── github.py
│   │   ├── parser.py
│   │   └── run.py
│   │
│   └── retrieval/
│       ├── embeddings.py
│       ├── index_repo.py
│       ├── indexer.py
│       └── vector_db.py
│
├── data/
│   ├── repos/
│   └── index/
│
├── tests/
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

---

## How It Works

### 1. Repository Ingestion

A GitHub repository is cloned locally and its source files are discovered.

The ingestion layer filters unsupported files and ignores directories such as:

- `.git`
- virtual environments
- `node_modules`
- build directories
- test directories
- IDE metadata

---

### 2. Code Parsing

Python source files are parsed using Python's `ast` module.

Instead of embedding entire files, the system extracts meaningful code units such as:

- functions
- asynchronous functions
- class methods
- asynchronous methods

Each chunk retains metadata including:

```text
file
symbol
symbol_type
parent
start_line
end_line
code
```

This allows retrieved results to retain their original source location.

---

### 3. Embedding Generation

Each code chunk is converted into a vector representation using:

```text
BAAI/bge-small-en-v1.5
```

The embeddings are normalized and stored in Qdrant.

---

### 4. Vector Indexing

Qdrant provides persistent local vector storage.

Each stored vector contains:

```text
Embedding
    +
Source code
    +
File metadata
    +
Symbol metadata
    +
Line information
```

This allows the system to retrieve both semantic content and the original source location.

---

### 5. Dense Semantic Retrieval

When a developer asks a question, the question is embedded using the same embedding model.

The resulting vector is compared against the indexed code chunks using cosine similarity.

The highest-ranking chunks are returned as the repository context.

---

### 6. RAG Generation

The retrieved code is passed to an LLM together with instructions to:

- answer using the supplied repository context
- avoid inventing repository behavior
- explain the implementation clearly
- cite relevant retrieved sources

The result is a grounded answer with citations such as:

```text
[1] requests/api.py:74-87
[2] requests/sessions.py:655-671
```

---

## API

The application exposes a FastAPI REST API.

### Health Check

```http
GET /health
```

Response:

```json
{
  "status": "ok"
}
```

### Query Repository

```http
POST /query
```

Request:

```json
{
  "question": "How does the application make an HTTP request?",
  "top_k": 5
}
```

Response:

```json
{
  "answer": "..."
}
```

The generated answer contains the explanation and relevant source citations.

---

## Running Locally

### 1. Clone the Repository

```bash
git clone https://github.com/malarms/github-codebase-rag.git
cd github-codebase-rag
```

### 2. Create the Virtual Environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the LLM

Create a `.env` file:

```env
LLM_BASE_URL=<your_openai_compatible_endpoint>
LLM_API_KEY=<your_api_key>
LLM_MODEL=<your_model>
```

The API layer uses an OpenAI-compatible client, allowing the underlying LLM provider to be configured independently of the RAG pipeline.

### 5. Index a Repository

Configure the repository ingestion/indexing process and run:

```bash
python -m app.retrieval.index_repo
```

This generates the persistent Qdrant index.

### 6. Start the API

```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

The API will be available at:

```text
http://localhost:8000
```

FastAPI's interactive documentation is available at:

```text
http://localhost:8000/docs
```

---

## Docker

The application can also be run using Docker.

Build the image:

```bash
docker build -t github-codebase-rag .
```

Run the container:

```bash
docker run -p 8000:8000 --env-file .env github-codebase-rag
```

Then check:

```text
GET http://localhost:8000/health
```

or query the assistant through:

```text
POST http://localhost:8000/query
```

---

## Evaluation

The retrieval pipeline was evaluated using a benchmark containing 9 repository-level questions with known relevant source locations.

### Results

| Metric | Result |
|---|---:|
| Recall@5 | **1.000** |
| MRR | **1.000** |

**Recall@5 = 1.000** means the relevant source was retrieved within the top five results for every benchmark query.

**MRR = 1.000** means the relevant source appeared at rank 1 for every benchmark query.

Example evaluation questions include:

- How does the web app execute a sandboxed program?
- Where is the POST API endpoint for executions implemented?
- How does the server check whether a request comes from the same origin?
- How is a Java snapshot created?
- How does the Python sandbox runner start a process?
- How does the application make an HTTP request?
- Where is the POST handler defined?
- How does the application validate incoming execution requests?
- Where are execution results returned to the client?

---

## Current Scope

The current implementation focuses on semantic retrieval of source-code chunks and RAG-based question answering.

The indexing pipeline supports multiple source-file extensions, while the current code-aware AST chunking implementation is primarily focused on Python source structure.

---

## Limitations

- Repository indexing is currently performed locally.
- The current code-aware AST parsing is Python-focused.
- The vector database is configured for local persistent storage.
- Retrieval quality depends on the embedding model and quality of the indexed code chunks.
- The system currently answers questions using retrieved repository context rather than maintaining conversational memory across queries.

---

## Future Improvements

Potential extensions include:

- Tree-sitter based parsing for deeper multi-language AST support
- Incremental repository indexing
- Commit-aware code indexing
- Symbol and dependency graph retrieval
- Repository-level dependency analysis
- Streaming responses
- Authentication and rate limiting
- Cloud vector database deployment
- Larger-scale evaluation benchmarks
