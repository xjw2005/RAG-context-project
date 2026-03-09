# api/routes.py

from fastapi import APIRouter

from app.schemas import Citation, IngestRequest, QueryRequest, QueryResponse
from app.dependencies import rag_engine


router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok"}

@router.post("/ingest")
def ingest(req: IngestRequest):
    rag_engine.ingest_document(req.doc_id, req.text)
    return {"message": "ingested", "doc_id": req.doc_id}

@router.post("/ingest-multi-vector")
def ingest_multi_vector(req: IngestRequest):
    rag_engine.ingest_document_multi_vector(req.doc_id, req.text)
    return {"message": "multi-vector ingested", "doc_id": req.doc_id}


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    # contexts = rag.retrieve(req.question, req.top_k)
    contexts = rag_engine.retrieve_combined(req.question, req.top_k)
    answer = rag_engine.generate_answer(req.question, contexts)
    citations = [Citation(doc_id=tuple[0], snippet=tuple[1]) for tuple in contexts]
    return QueryResponse(answer=answer, citations=citations)

@router.post("/query-multi-vector", response_model=QueryResponse)
def query_multi_vector(req: QueryRequest):
    contexts = rag_engine.strategies["multi_vector"].retrieve(req.question, req.top_k)
    answer = rag_engine.generate_answer(req.question, contexts)
    citations = [Citation(doc_id=tuple[0], snippet=tuple[1]) for tuple in contexts]
    return QueryResponse(answer=answer, citations=citations)
