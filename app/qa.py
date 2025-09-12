import os, textwrap
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

from chromadb import PersistentClient
from chromadb.config import Settings
from app.embed_openai import embed_texts
from app.llm_openai import chat

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DIR", "./vectorstore/aprobo_v1")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "aprobo_v1_e1536")

def _open_collection():
    client = PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)  # kill posthog/telemetry
    )
    return client.get_or_create_collection(name=COLLECTION_NAME)

def retrieve(query: str, k: int = 6) -> Dict[str, Any]:
    """Return raw Chroma results: ids, docs, metadatas, distances."""
    collection = _open_collection()
    qvec = embed_texts([query])[0]
    res = collection.query(
        query_embeddings=[qvec],
        n_results=k,
        include=["documents", "metadatas", "distances", "uris"]
    )
    # res keys -> "ids", "documents", "metadatas", "distances"
    return {k: v[0] for k, v in res.items() if isinstance(v, list)}  # unwrap single-query

def _format_context(docs: List[str], metas: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, str]]]:
    """
    Creates a readable context block and a list of sources with title/url.
    Each source entry: {"ref":"[1]", "title":"...", "url":"..."} aligned with doc index.
    """
    sources = []
    context_blocks = []
    for i, (doc, md) in enumerate(zip(docs, metas), start=1):
        title = (md.get("product_name") or md.get("source_title") or md.get("type") or "Source").strip()
        url = (md.get("url") or md.get("source_url") or "").strip()
        # compact doc text
        snippet = doc.strip()
        context_blocks.append(f"[{i}] {title}\n{snippet}\n")
        sources.append({"ref": f"[{i}]", "title": title, "url": url})
    return "\n".join(context_blocks), sources

SYSTEM_PROMPT = """\
You are a precise assistant for Aprobo’s product and acoustic-flooring knowledge.

Rules:
- Answer using ONLY the provided context.
- If the answer isn't in context, say you don't know and suggest what to check next.
- Keep answers concise, technical when needed, and include a short 'Citations' section listing source refs like [1], [2].
- Do not invent values/specs.
"""

USER_TEMPLATE = """\
Question:
{question}

Context:
{context}

Instructions:
1) Provide a direct answer first.
2) Then add a 'Citations' line listing the matching [n] refs you used.
3) If multiple options apply, compare them briefly.
"""

def answer_question(question: str, k: int = 6) -> Dict[str, Any]:
    hits = retrieve(question, k=k)
    docs = hits.get("documents", [])
    metas = hits.get("metadatas", [])
    distances = hits.get("distances", [])

    if not docs:
        return {"answer": "I don't have enough information to answer from the current index.",
                "sources": []}

    context, sources = _format_context(docs, metas)
    msg_user = USER_TEMPLATE.format(question=question, context=context)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": msg_user},
    ]
    answer = chat(messages)

    # Attach the clean source list with URLs for your UI
    return {
        "answer": answer,
        "sources": sources,
        "distances": distances
    }

# CLI entrypoint: python -m app.qa "your question"
if __name__ == "__main__":
    import sys, json
    q = " ".join(sys.argv[1:]).strip() or "What is Art Base Parquet and typical applications?"
    res = answer_question(q, k=6)
    print("\n" + res["answer"] + "\n")
    if res["sources"]:
        print("Sources:")
        for s in res["sources"]:
            u = f" {s['url']}" if s["url"] else ""
            print(f"  {s['ref']} {s['title']}{u}")
