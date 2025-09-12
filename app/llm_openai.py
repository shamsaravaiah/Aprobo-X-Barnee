import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Keep a single function so you can swap models centrally
def chat(messages: List[Dict[str, str]], model: str = "gpt-4o-mini") -> str:
    """
    messages = [{"role":"system","content":"..."}, {"role":"user","content":"..."}]
    returns the assistant's text content
    """
    resp = _client.chat.completions.create(model=model, messages=messages, temperature=0.2)
    return resp.choices[0].message.content.strip()
