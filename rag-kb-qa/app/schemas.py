from pydantic import BaseModel
from typing import List

class IngestRequest(BaseModel):
    doc_id: str
    text: str

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

class Citation(BaseModel):
    doc_id: str
    snippet: str

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
