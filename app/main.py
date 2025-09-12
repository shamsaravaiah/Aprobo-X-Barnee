from fastapi import FastAPI
from pydantic import BaseModel
from qa import answer_question

app = FastAPI(title="Aprobo QA API")

class QAReq(BaseModel):
    question: str
    k: int = 6

@app.post("/qa")
def qa(req: QAReq):
    return answer_question(req.question, k=req.k)

@app.get("/health")
def health():
    return {"status": "ok"}
