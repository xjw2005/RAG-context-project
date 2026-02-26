from fastapi import FastAPI
from app.schemas import IngestRequest, QueryRequest, QueryResponse, Citation
from app import rag

app = FastAPI(title="RAG KB QA MVP")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest")
def ingest(req: IngestRequest):
    rag.ingest(req.doc_id, req.text)
    return {"message": "ingested", "doc_id": req.doc_id}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    contexts = rag.retrieve(req.question, req.top_k)
    answer = rag.generate_answer(req.question, contexts)
    citations = [Citation(doc_id=d, snippet=s) for d, s in contexts]
    return QueryResponse(answer=answer, citations=citations)
