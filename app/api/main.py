from fastapi import FastAPI
from pydantic import BaseModel

from app.generation.rag import RAGPipeline


app = FastAPI(
    title="AI GitHub Codebase Assistant",
    description="RAG-based codebase question answering API",
    version="1.0.0",
)


rag = RAGPipeline()


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    answer = rag.ask(
        question=request.question,
        top_k=request.top_k,
    )

    return {
        "answer": answer
    }
