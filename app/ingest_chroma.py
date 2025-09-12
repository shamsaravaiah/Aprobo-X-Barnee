import os, json, glob, re
from collections import Counter
from dotenv import load_dotenv
from tqdm import tqdm

import chromadb
from chromadb.config import Settings

import vertexai
from vertexai.language_models import TextEmbeddingModel

load_dotenv()
PROJECT_ID     = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION       = os.environ.get("GOOGLE_CLOUD_LOCATION", "europe-west1")
CHROMA_PATH    = os.environ.get("CHROMA_PATH", ".chroma")
COLLECTION_NAME= os.environ.get("COLLECTION_NAME", "aprobo-seed")
DATA_DIR       = os.environ.get("DATA_DIR", "data")

assert PROJECT_ID, "Set GOOGLE_CLOUD_PROJECT in .env"

vertexai.init(project=PROJECT_ID, location=LOCATION)
embed_model = TextEmbeddingModel.from_pretrained("text-embedding-004")

client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(allow_reset=False))
collection = client.get_or_create_collection(name=COLLECTION_NAME)

def read_jsonl_files(folder):
    for fp in sorted(glob.glob(os.path.join(folder, "*.jsonl"))):
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line: 
                    continue
                try:
                    obj=json.loads(line)
                    yield fp, obj
                except json.JSONDecodeError:
                    continue

# --- light cleaner to remove nav spam/dup lines and normalize whitespace ---
NAV_PATTERNS = [
    r"HomeProducts.*?Sök",             # site menu chunk (greedy but non-DOTALL)
    r"Aprobo AB.*?Sweden",             # repeated footer/contact block
]
def clean_text(t: str) -> str:
    if not t: 
        return ""
    # Remove obvious site-navigation/footer spam
    for pat in NAV_PATTERNS:
        t = re.sub(pat, " ", t, flags=re.IGNORECASE|re.DOTALL)
    # Collapse repeated sentences/phrases (very simple heuristic)
    # keep order while removing exact duplicate lines
    lines = [l.strip() for l in re.split(r"[\r\n]+", t)]
    seen = set()
    uniq_lines = []
    for l in lines:
        if not l: 
            continue
        if l.lower() in seen:
            continue
        seen.add(l.lower())
        uniq_lines.append(l)
    t = " ".join(uniq_lines)
    # Normalize whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t

def chunkify(text, max_chars=1800, overlap=200):
    text = text.strip()
    n = len(text)
    if n <= max_chars:
        return [text] if text else []
    chunks = []
    start = 0
    while start < n:
        end = min(n, start + max_chars)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n: break
        start = max(0, end - overlap)
    return chunks

def embed_batch(texts):
    embs = embed_model.get_embeddings(texts)
    return [e.values for e in embs]

def get_existing_ids():
    # pull many; OK for small/medium corpora
    res = collection.get(limit=1_000_000)
    return set(res.get("ids", []))

def main():
    print(f"Project: {PROJECT_ID}  Location: {LOCATION}")
    print(f"Chroma: {CHROMA_PATH}  Collection: {COLLECTION_NAME}")
    print(f"Data dir: {DATA_DIR}")

    existing = get_existing_ids()
    BATCH_SIZE = 32
    pend_docs, pend_ids, pend_meta = [], [], []
    added = 0
    found_files = False

    for fp, obj in read_jsonl_files(DATA_DIR):
        found_files = True
        doc_id = str(obj.get("id") or "")
        raw_text = obj.get("text") or ""
        meta = obj.get("metadata") or {}

        if not doc_id or not raw_text:
            continue

        # one upstream record may represent a sub-section (type: main_description/applications/features)
        # We keep the original id and also suffix chunks.
        # Skip if the exact doc-id already present (idempotency for re-runs).
        if doc_id in existing:
            continue

        text = clean_text(raw_text)
        chunks = chunkify(text)
        if not chunks:
            continue

        for i, ch in enumerate(chunks):
            cid = f"{doc_id}::chunk{i:03d}"
            pend_ids.append(cid)
            pend_docs.append(ch)
            # carry all provided metadata + helpful fields
            m = {
                "source_id": doc_id,
                "chunk_index": i,
                "num_chunks": len(chunks),
                "ingest_file": os.path.basename(fp),
            }
            # propagate known fields from your schema
            for k in ["product_id","product_name","category","subcategory","url","type"]:
                if k in meta:
                    m[k] = meta[k]
            pend_meta.append(m)

            if len(pend_docs) >= BATCH_SIZE:
                vectors = embed_batch(pend_docs)
                collection.add(ids=pend_ids, documents=pend_docs, metadatas=pend_meta, embeddings=vectors)
                added += len(pend_docs)
                pend_docs, pend_ids, pend_meta = [], [], []

    if not found_files:
        print("No .jsonl files found in ./data")
        return

    if pend_docs:
        vectors = embed_batch(pend_docs)
        collection.add(ids=pend_ids, documents=pend_docs, metadatas=pend_meta, embeddings=vectors)
        added += len(pend_docs)

    print(f"✅ Ingestion complete. Added {added} chunks. Collection size: {collection.count()}")

if __name__ == "__main__":
    main()
