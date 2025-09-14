from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.qa import answer_question
import os

app = FastAPI(title="Aprobo QA API")

# Configure CORS the same way as in app/api.py
# Read comma-separated origins from environment
allowed_raw = os.getenv(
    "ALLOWED_ORIGINS",
    "https://aprobo-chat-buddy.lovable.app,https://id-preview--a79b81c8-b9da-4acc-abb4-5e6bb7dbc57a.lovable.app",
)
allowed = [o.strip() for o in allowed_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "*"],
    expose_headers=["Content-Length", "Content-Type"],
    max_age=86400,
)

class QAReq(BaseModel):
    question: str
    k: int = 6

@app.post("/qa")
def qa(req: QAReq):
    return answer_question(req.question, k=req.k)

@app.get("/health")
def health():
    return {"status": "ok"}
