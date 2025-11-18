import os
import json
import re
import hashlib
import datetime
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np

# Lazy import of optional heavy dependencies
faiss = None
try:
    import faiss
except ImportError:
    faiss = None

# --------------------------------------------------------------------------- #
# Configuration (edit or override via CLI/environment)
# --------------------------------------------------------------------------- #
DATA_DIR = Path("data")
OUTPUT_DIR = Path("knowledge_base")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Example input files (replace or extend with your own English texts)
REGULATION_FILES = [
    # DATA_DIR / "regulations_part1.txt",
    # DATA_DIR / "regulations_part2.txt",
    # DATA_DIR / "interview_transcripts.txt",
]

INCIDENT_JSON = DATA_DIR / "sample_incidents.json"   # Public NTSB/ASRS-style records

# Chunking parameters
CHUNK_SIZE_CHARS = 1000
CHUNK_OVERLAP = 200

# Embedding options
DEFAULT_LOCAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DASHSCOPE_EMBED_MODEL = "text-embedding-v2"

# FAISS index location (must be writable; adjust if needed)
FAISS_INDEX_PATH = Path("faiss_index") / "kb_faiss.index"
FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def normalize_text(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("—", "-").replace("–", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, size: int, overlap: int) -> List[Tuple[int, int, str]]:
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(length, start + size)
        chunks.append((start, end, text[start:end]))
        if end == length:
            break
        start = end - overlap

    return chunks


def load_incident_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    records = []

    for idx, item in enumerate(data):
        # Prioritise narrative fields
        candidates = [
            item.get("AnalysisNarrative"), item.get("FactualNarrative"),
            item.get("PrelimNarrative"), item.get("ProbableCause"),
            item.get("Summary"), item.get("Synopsis"), item.get("Narrative")
        ]
        narrative = " ".join(p for p in candidates if isinstance(p, str) and p.strip())

        if not narrative.strip():
            # Fallback to key fields
            parts = []
            for key in ["NtsbNumber", "City", "Country", "EventDate", "EventType"]:
                val = item.get(key)
                if val:
                    parts.append(f"{key}: {val}")
            narrative = " | ".join(parts)

        records.append({
            "doc_id": f"incident_{idx:06d}",
            "text": normalize_text(narrative),
            "meta": {k: item.get(k) for k in ["NtsbNumber", "City", "Country", "EventDate", "EventType"]}
        })

    return records


def document_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Corpus construction
# --------------------------------------------------------------------------- #
def build_corpus() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    docstore: Dict[str, Any] = {}

    # 1. Regulatory / interview texts
    for path in REGULATION_FILES:
        if not path.exists():
            print(f"[Warning] Regulation file not found: {path}")
            continue

        raw = normalize_text(read_text_file(path))
        doc_id = f"regulation_{document_hash(raw)}"
        spans = chunk_text(raw, CHUNK_SIZE_CHARS, CHUNK_OVERLAP)

        docstore[doc_id] = {
            "source": path.name,
            "side": "institution",
            "path": str(path),
            "characters": len(raw),
            "chunks": len(spans)
        }

        for chunk_idx, (s, e, txt) in enumerate(spans):
            chunk_id = f"{doc_id}_c{chunk_idx:04d}"
            chunks.append({
                "id": chunk_id,
                "text": txt,
                "source": path.name,
                "side": "institution",
                "doc_id": doc_id,
                "chunk_idx": chunk_idx,
                "char_range": [s, e]
            })

    # 2. Incident reports (external practice)
    incidents = load_incident_records(INCIDENT_JSON)
    for rec in incidents:
        raw = rec["text"]
        doc_id = rec["doc_id"]
        spans = chunk_text(raw, CHUNK_SIZE_CHARS, CHUNK_OVERLAP)

        docstore[doc_id] = {
            "source": "sample_incidents.json",
            "side": "external_practice",
            "path": str(INCIDENT_JSON),
            "characters": len(raw),
            "chunks": len(spans),
            **rec["meta"]
        }

        for chunk_idx, (s, e, txt) in enumerate(spans):
            chunk_id = f"{doc_id}_c{chunk_idx:04d}"
            chunks.append({
                "id": chunk_id,
                "text": txt,
                "source": "sample_incidents.json",
                "side": "external_practice",
                "doc_id": doc_id,
                "chunk_idx": chunk_idx,
                "char_range": [s, e]
            })

    return chunks, docstore


# --------------------------------------------------------------------------- #
# Embedding functions
# --------------------------------------------------------------------------- #
def embed_dashscope(texts: List[str], api_key: str, api_base: str, model: str) -> np.ndarray:
    import requests

    url = f"{api_base.rstrip('/')}/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    vectors = []
    batch_size = 64

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        payload = {"model": model, "input": batch}
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()["data"]
        data.sort(key=lambda x: x["index"])
        vectors.extend([item["embedding"] for item in data])

    return np.array(vectors, dtype=np.float32)


def embed_local(texts: List[str], model_name: str = DEFAULT_LOCAL_MODEL) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False
    )
    return embeddings.astype(np.float32)


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


# --------------------------------------------------------------------------- #
# FAISS index construction
# --------------------------------------------------------------------------- #
def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    if faiss is None:
        raise ImportError("faiss not available. Install with: pip install faiss-cpu")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    normalized = l2_normalize(embeddings)
    index.add(normalized)
    return index


# --------------------------------------------------------------------------- #
# Main execution
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Build aviation Accimap knowledge base")
    parser.add_argument("--use-dashscope", action="store_true", help="Use DashScope embedding API")
    parser.add_argument("--api-key", type=str, default=os.getenv("DASHSCOPE_API_KEY", ""), help="DashScope API key")
    parser.add_argument("--api-base", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--local-model", type=str, default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--skip-cache", action="store_true", help="Force re-computation of embeddings")
    parser.add_argument("--test-query", type=str, default="emergency response communication failure resource dispatch")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if args.use_dashscope and not args.api_key:
        raise ValueError("DashScope API key required when --use-dashscope is set")

    # 1. Corpus
    chunks, docstore = build_corpus()
    if not chunks:
        raise RuntimeError("No chunks created – check input files")

    # Persist chunk metadata
    meta_path = OUTPUT_DIR / "kb_meta.jsonl"
    with open(meta_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    with open(OUTPUT_DIR / "kb_docstore.json", "w", encoding="utf-8") as f:
        json.dump(docstore, f, ensure_ascii=False, indent=2)

    # 2. Embeddings
    texts = [c["text"] for c in chunks]
    emb_cache = OUTPUT_DIR / "kb_embeddings.npy"

    if not args.skip_cache and emb_cache.exists():
        embeddings = np.load(emb_cache)
        print(f"Loaded cached embeddings from {emb_cache}")
    else:
        print(f"Generating embeddings ({'DashScope' if args.use_dashscope else 'local'})...")
        if args.use_dashscope:
            embeddings = embed_dashscope(texts, args.api_key, args.api_base, DASHSCOPE_EMBED_MODEL)
        else:
            embeddings = embed_local(texts, args.local_model)
        np.save(emb_cache, embeddings)

    # 3. FAISS index
    index = build_faiss_index(embeddings)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"FAISS index saved to {FAISS_INDEX_PATH}")

    # 4. Report & demo retrieval
    report = {
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "total_chunks": len(chunks),
        "embedding_model": DASHSCOPE_EMBED_MODEL if args.use_dashscope else args.local_model,
        "dimension": embeddings.shape[1],
        "artifacts": {
            "meta": str(meta_path),
            "docstore": str(OUTPUT_DIR / "kb_docstore.json"),
            "embeddings": str(emb_cache),
            "faiss_index": str(FAISS_INDEX_PATH)
        }
    }
    with open(OUTPUT_DIR / "knowledge_store_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Simple retrieval demo
    if args.test_query:
        if args.use_dashscope:
            query_vec = embed_dashscope([args.test_query], args.api_key, args.api_base, DASHSCOPE_EMBED_MODEL)
        else:
            query_vec = embed_local([args.test_query], args.local_model)
        query_vec = l2_normalize(query_vec)

        distances, indices = index.search(query_vec, args.top_k)
        demo_lines = []
        for rank, (score, idx) in enumerate(zip(distances[0], indices[0]), 1):
            chunk = chunks[idx]
            preview = chunk["text"].replace("\n", " ")[:240]
            demo_lines.append(f"[{rank}] score={score:.4f} | {chunk['side']} | {preview}...")

        demo_path = OUTPUT_DIR / "search_demo.txt"
        demo_path.write_text("\n".join(demo_lines), encoding="utf-8")
        print(f"Retrieval demo written to {demo_path}")

    print("Knowledge base construction complete.")


if __name__ == "__main__":
    main()