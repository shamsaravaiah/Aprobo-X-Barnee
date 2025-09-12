# Minimal OpenAI embedding helper
import os
from dotenv import load_dotenv
from typing import List
from openai import OpenAI

load_dotenv()

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> List[list]:
    # Returns a list of vectors (list[float]) matching input order
    resp = _client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]
