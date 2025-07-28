# bm25_utils.py
import os
import pickle
from rank_bm25 import BM25Okapi

BASE_DIR = "bm25_indexes"

def load_bm25(user_id: str):
    """
    Returns (bm25, chunks) or (None, []) if none exists.
    """
    path = os.path.join(BASE_DIR, f"{user_id}_bm25.pkl")
    if not os.path.exists(path):
        return None, []
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["chunks"]

def build_and_persist_bm25(user_id: str, new_chunks: list[dict]) -> BM25Okapi:
    """
    Build (or update) a BM25 index by appending `new_chunks` to any existing ones.
    Deduplicates by (source_document, chunk_index), then rebuilds and persists.
    """
    os.makedirs(BASE_DIR, exist_ok=True)
    path = os.path.join(BASE_DIR, f"{user_id}_bm25.pkl")

    # 1. Load old chunks if they exist
    _, old_chunks = load_bm25(user_id)

    # 2. Merge + dedupe
    combined = []
    seen = set()
    for chunk in old_chunks + new_chunks:
        # Use document name + chunk index as a unique key
        key = f"{chunk['source_document']}#{chunk['chunk_index']}"
        if key not in seen:
            seen.add(key)
            combined.append(chunk)

    # 3. Build BM25 on the full, deduped list
    docs = [c["text"] for c in combined]
    tokenized = [d.lower().split() for d in docs]
    bm25 = BM25Okapi(tokenized)

    # 4. Persist both the new index and the full chunk list
    with open(path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": combined}, f)

    return bm25

def delete_user_bm25_index(user_id: str):
    """
    Remove the persisted BM25 index for the given user, if it exists.
    """
    path = os.path.join(BASE_DIR, f"{user_id}_bm25.pkl")
    try:
        os.remove(path)
        print(f"🗑️  Deleted BM25 index file: {path}")
    except FileNotFoundError:
        print(f"⚠️  No BM25 index to delete at: {path}")
