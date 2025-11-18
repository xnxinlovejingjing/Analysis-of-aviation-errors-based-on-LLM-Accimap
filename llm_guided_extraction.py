import os
import json
import argparse
import datetime
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import faiss
import requests

# --------------------------------------------------------------------------- #
# Paths (relative – adjust only if you move the repository structure)
# --------------------------------------------------------------------------- #
KB_DIR = Path("knowledge_base")
KB_META = KB_DIR / "kb_meta.jsonl"
KB_EMBEDDINGS = KB_DIR / "kb_embeddings.npy"
FAISS_INDEX_PATH = Path("faiss_index") / "kb_faiss.index"

OUTPUT_DIR = Path("extraction_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# LLM configuration
# --------------------------------------------------------------------------- #
API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
CHAT_MODEL = "qwen-plus"

TEMPERATURE = 0.10
TOP_P = 0.95
MAX_TOKENS = 1400

TOPK_TOTAL = 6          # Total evidence chunks fed to the model
CROSS_SIDE_K = 3        # Balance between same-side and opposite-side evidence

# --------------------------------------------------------------------------- #
# Accimap canonical mapping (normalises variations from the LLM)
# --------------------------------------------------------------------------- #
LEVEL_CANONICAL = {
    "regulator": "Regulator", "authority": "Regulator", "caac": "Regulator", "faa": "Regulator", "easa": "Regulator",
    "airline": "Airline", "company": "Airline", "operator": "Airline",
    "management": "Mgmt", "manager": "Mgmt", "ops_management": "Mgmt",
    "crew": "Crew", "pilot": "Crew", "dispatcher": "Crew", "atc": "Crew",
    "equipment": "Equipment", "aircraft": "Equipment", "system": "Equipment",
    "government": "Government", "ministry": "Government"
}

TYPE_CANONICAL = {
    "policy_gap": "policy_gap", "regulatory_gap": "policy_gap",
    "resource_delay": "resource_delay", "resourcing_delay": "resource_delay",
    "training_insufficiency": "training_insufficiency", "insufficient_training": "training_insufficiency",
    "communication_breakdown": "communication_breakdown", "comms_breakdown": "communication_breakdown",
    "tech_failure": "tech_failure", "technical_failure": "tech_failure",
    "procedure_mismatch": "procedure_mismatch", "sop_mismatch": "procedure_mismatch",
    "monitoring_lapse": "monitoring_lapse", "oversight_gap": "monitoring_lapse",
    "data_issue": "data_issue", "data_quality": "data_issue",
    "other": "other"
}

# --------------------------------------------------------------------------- #
# Prompt templates (single-turn, strict JSON output)
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """
You are an aviation safety analyst performing Accimap-based attribution of emergency-response failures.
Extract ONLY what is explicitly supported by the provided TARGET and EVIDENCE chunks.
Return valid JSON only – no explanations, no markdown, no extra fields.
If evidence is insufficient for a factor/relation, omit it rather than speculate.
Confidence must reflect evidential clarity (0.0–1.0).
"""

USER_TEMPLATE = """
[TARGET CHUNK]
{target_text}

[EVIDENCE CHUNKS]
{evidence_text}

Return JSON conforming exactly to this schema:
{
  "factors": [
    {
      "text_span": "verbatim text from TARGET or EVIDENCE",
      "level": "Regulator|Airline|Mgmt|Crew|Equipment|Government",
      "type": "policy_gap|resource_delay|training_insufficiency|communication_breakdown|tech_failure|procedure_mismatch|monitoring_lapse|data_issue|other",
      "polarity": "cause|consequence",
      "char_idx": [start, end],           // relative to TARGET chunk; use [-1,-1] if span is only in EVIDENCE
      "confidence": 0.85
    }
  ],
  "relations": [
    {
      "source": 0,
      "target": 1,
      "relation": "causes|enables|exacerbates",
      "evidence_span": "verbatim supporting text",
      "char_idx": [start, end],           // relative to TARGET or [-1,-1]
      "confidence": 0.90
    }
  ],
  "attribution": {
    "axis": "design_flaw|execution_gap|mixed|uncertain",
    "rationale_span": "short justification from context",
    "char_idx": [start, end],
    "confidence": 0.80
  }
}
"""

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def load_chunks() -> List[Dict[str, Any]]:
    chunks = []
    with open(KB_META, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def load_embeddings() -> np.ndarray:
    return np.load(KB_EMBEDDINGS)


def load_faiss_index() -> faiss.Index:
    return faiss.read_index(str(FAISS_INDEX_PATH))


def embed_texts(texts: List[str], api_key: str) -> np.ndarray:
    url = f"{API_BASE.rstrip('/')}/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    vectors = []

    for i in range(0, len(texts), 64):
        batch = texts[i:i+64]
        payload = {"model": "text-embedding-v2", "input": batch}
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda x: x["index"])
        vectors.extend([item["embedding"] for item in data])

    return np.array(vectors, dtype=np.float32)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def retrieve_evidence(target_chunk: Dict[str, Any],
                      chunks: List[Dict[str, Any]],
                      embeddings: np.ndarray,
                      index: faiss.Index,
                      embed_fn: callable,
                      topk: int = TOPK_TOTAL) -> List[Dict[str, Any]]:
    query_text = target_chunk["text"]
    query_vec = embed_fn([query_text])[0:1]
    query_vec = normalize_vectors(query_vec)

    distances, indices = index.search(query_vec, topk * 3)  # overshoot to allow side balancing

    same_side = []
    cross_side = []

    target_side = target_chunk["side"]

    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        candidate = chunks[idx]
        if candidate["id"] == target_chunk["id"]:
            continue
        if candidate["side"] == target_side:
            same_side.append((dist, candidate))
        else:
            cross_side.append((dist, candidate))

    # Balance retrieval
    selected = []
    selected.extend([c for _, c in sorted(same_side)[:CROSS_SIDE_K]])
    selected.extend([c for _, c in sorted(cross_side)[:CROSS_SIDE_K]])
    selected.extend([c for _, c in sorted(same_side)[CROSS_SIDE_K:CROSS_SIDE_K + (topk - CROSS_SIDE_K * 2)]])

    return selected[:topk]


def call_llm(messages: List[Dict[str, str]], api_key: str) -> str:
    url = f"{API_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=180)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def parse_llm_output(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end == 0:
        return {"factors": [], "relations": [], "attribution": {}}
    try:
        return json.loads(raw[start:end])
    except json.JSONDecodeError:
        return {"factors": [], "relations": [], "attribution": {}}


def canonicalise(output: Dict[str, Any]) -> Dict[str, Any]:
    for f in output.get("factors", []):
        level = f.get("level", "").lower()
        f["level"] = LEVEL_CANONICAL.get(level, "Crew")
        typ = f.get("type", "").lower()
        f["type"] = TYPE_CANONICAL.get(typ, "other")

    attr = output.get("attribution", {})
    axis = attr.get("axis", "").lower()
    if axis not in {"design_flaw", "execution_gap", "mixed", "uncertain"}:
        attr["axis"] = "uncertain"

    return output


# --------------------------------------------------------------------------- #
# Main extraction loop
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", type=str, required=True, help="DashScope API key")
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM calls (for pipeline testing)")
    args = parser.parse_args()

    chunks = load_chunks()
    embeddings = load_embeddings()
    index = load_faiss_index()

    def embed_fn(texts): return embed_texts(texts, args.api_key)

    factors_out = []
    relations_out = []
    attributions_out = []

    for i, target in enumerate(chunks):
        evidence_chunks = retrieve_evidence(target, chunks, embeddings, index, embed_fn)

        evidence_text = "\n\n".join(
            f"[{j+1}] ({c['side']}) {c['text'][:1500]}"
            for j, c in enumerate(evidence_chunks)
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                target_text=target["text"],
                evidence_text=evidence_text if evidence_text else "No additional evidence retrieved."
            )}
        ]

        if args.dry_run:
            raw_output = '{"factors": [], "relations": [], "attribution": {}}'
        else:
            raw_output = call_llm(messages, args.api_key)

        parsed = parse_llm_output(raw_output)
        parsed = canonicalise(parsed)

        # Normalise and store factors
        for j, f in enumerate(parsed.get("factors", [])):
            factors_out.append({
                "chunk_uid": target["id"],
                "doc_id": target["doc_id"],
                "source": target["source"],
                "side": target["side"],
                "factor_idx": j,
                "text_span": f.get("text_span", "").strip(),
                "level": f.get("level"),
                "type": f.get("type"),
                "polarity": f.get("polarity", "cause"),
                "char_idx": f.get("char_idx", [-1, -1]),
                "confidence": float(f.get("confidence", 0.0))
            })

        # Normalise and store relations
        for r in parsed.get("relations", []):
            relations_out.append({
                "chunk_uid": target["id"],
                "doc_id": target["doc_id"],
                "source": target["source"],
                "side": target["side"],
                "source_idx": int(r.get("source", -1)),
                "target_idx": int(r.get("target", -1)),
                "relation": r.get("relation", "causes"),
                "evidence_span": r.get("evidence_span", "").strip(),
                "char_idx": r.get("char_idx", [-1, -1]),
                "confidence": float(r.get("confidence", 0.0))
            })

        # Attribution (one per chunk)
        att = parsed.get("attribution", {}) or {}
        attributions_out.append({
            "chunk_uid": target["id"],
            "doc_id": target["doc_id"],
            "source": target["source"],
            "side": target["side"],
            "axis": att.get("axis", "uncertain"),
            "rationale_span": att.get("rationale_span", "").strip(),
            "char_idx": att.get("char_idx", [-1, -1]),
            "confidence": float(att.get("confidence", 0.0))
        })

        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{len(chunks)} chunks")

    # Persist raw results
    (OUTPUT_DIR / "factors.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in factors_out), encoding="utf-8")
    (OUTPUT_DIR / "relations.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in relations_out), encoding="utf-8")
    (OUTPUT_DIR / "attributions.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in attributions_out), encoding="utf-8")

    # Convenience CSVs
    import pandas as pd
    pd.DataFrame(factors_out).to_csv(OUTPUT_DIR / "accimap_factors.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(relations_out).to_csv(OUTPUT_DIR / "accimap_relations.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(attributions_out).to_csv(OUTPUT_DIR / "attributions.csv", index=False, encoding="utf-8-sig")

    print(f"Extraction complete. Results saved to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()