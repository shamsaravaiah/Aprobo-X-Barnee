# Ingest Aprobo JSONL seeds into a persistent Chroma collection.
# Expects JSONL lines like: {"id": "...", "text": "...", "metadata": {...}}

import os, glob, json, hashlib
from typing import List, Dict, Any, Tuple

from chromadb import PersistentClient
from dotenv import load_dotenv
from chromadb.config import Settings
from embed_openai import embed_texts

load_dotenv()

CHROMA_DIR      = os.getenv("CHROMA_DIR", "./vectorstore/aprobo_v1")
SEED_GLOB       = os.getenv("SEED_GLOB", "data/*.jsonl")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "aprobo_v1_e1536")

def sha16(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

def load_jsonl_files(pattern: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No JSONL files matched: {pattern}")
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception as e:
                    raise ValueError(f"{fp}:{i} invalid JSON: {e}\nLine: {line[:200]}")
                records.append(obj)
    if not records:
        raise ValueError("No records found across JSONL files.")
    return records

def normalize_item(it: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    # Aprobo schema: require id & text; metadata optional
    text = (it.get("text") or "").strip()
    if not text:
        raise ValueError("Missing 'text' in record.")
    doc_id = (it.get("id") or "").strip() or sha16(text[:256])
    md = it.get("metadata") or {}
    # Ensure metadata is JSON-serializable scalars
    safe_meta = {}
    for k, v in md.items():
        if isinstance(v, (list, dict)):
            safe_meta[k] = json.dumps(v, ensure_ascii=False)
        else:
            safe_meta[k] = v
    return text, doc_id, safe_meta

def main():
    print(f"Loading seeds from: {SEED_GLOB}")
    items = load_jsonl_files(SEED_GLOB)
    print(f"Loaded {len(items)} raw records.")

    texts, ids, metas = [], [], []
    for it in items:
        try:
            t, id_, md = normalize_item(it)
        except Exception as e:
            # Skip bad records but show why
            print(f"Skipping record due to error: {e}")
            continue
        texts.append(t); ids.append(id_); metas.append(md)

    if not texts:
        raise ValueError("No valid items to ingest after normalization.")

    print(f"Embedding {len(texts)} texts…")
    embeddings = embed_texts(texts, model="text-embedding-3-small")

    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)  # set False to avoid posthog import
    )
    col = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    print(f"Upserting {len(ids)} docs into '{COLLECTION_NAME}'…")
    col.upsert(documents=texts, ids=ids, metadatas=metas, embeddings=embeddings)

    print(f"✅ Ingest complete. Count = {col.count()} (dir: {CHROMA_DIR})")

if __name__ == "__main__":
    main()
