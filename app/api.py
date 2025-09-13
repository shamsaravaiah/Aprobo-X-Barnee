from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

from app.qa import answer_question, retrieve  # uses your existing code

app = FastAPI(title="Aprobo QA (local)")



allowed = os.getenv("ALLOWED_ORIGINS", "https://aprobo-chat-buddy.lovable.app/")
origins = [o.strip() for o in allowed.split(",") if o.strip()]

# CORS for quick local testing (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else ["*"],  # use specific origins in prod
    allow_credentials=True,                        # set to False if you don’t need cookies/auth
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "*"],
    expose_headers=["Content-Length", "Content-Type"],
    max_age=86400,
)

class QAReq(BaseModel):
    question: str
    k: int = 6

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/qa")
def qa(req: QAReq):
    res = answer_question(req.question, k=req.k)
    if not res:
        raise HTTPException(status_code=500, detail="LLM error")
    return res

# Optional: raw search endpoint to inspect retrieval without LLM
class SearchReq(BaseModel):
    query: str
    k: int = 6

@app.post("/search")
def search(req: SearchReq):
    hits = retrieve(req.query, k=req.k)
    return hits
