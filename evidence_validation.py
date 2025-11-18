import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).parent.resolve()
EXTRACTION_DIR = BASE_DIR / "extraction_results"
KB_META_PATH = BASE_DIR / "knowledge_base" / "kb_meta.jsonl"

HIGH_CONF_DIR = EXTRACTION_DIR / "high_confidence"
REVIEW_DIR = EXTRACTION_DIR / "review_queue"
REPORT_PATH = EXTRACTION_DIR / "validation_report.json"

HIGH_CONF_DIR.mkdir(parents=True, exist_ok=True)
REVIEW_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Validation thresholds
# --------------------------------------------------------------------------- #
CONFIDENCE_THRESHOLD_FACTOR = 0.50
CONFIDENCE_THRESHOLD_RELATION = 0.50
CONFIDENCE_THRESHOLD_ATTRIB = 0.50

TOKEN_SIMILARITY_THRESHOLD = 0.85   # Jaccard on alphanumeric tokens

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
    path.write_text(text + "\n", encoding="utf-8")


def simple_tokens(text: str) -> set:
    return set(re.findall(r"[A-Za-z0-9]+", text.lower()))


def token_jaccard(a: str, b: str) -> float:
    tokens_a = simple_tokens(a)
    tokens_b = simple_tokens(b)
    if not tokens_a and not tokens_b:
        return 1.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union > 0 else 0.0


def load_chunk_text() -> Dict[str, str]:
    """Map chunk_uid → full text."""
    chunk_map = {}
    for line in KB_META_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        chunk_id = obj.get("id")
        if chunk_id:
            chunk_map[chunk_id] = obj.get("text", "")
    return chunk_map


def clip_span(start: int, end: int, length: int) -> Tuple[int, int]:
    if start < 0 or end < 0:
        return -1, -1
    s = max(0, min(start, length))
    e = max(0, min(end, length))
    return s, e if s <= e else (e, s)


def validate_span(item_span: str,
                   char_idx: List[int],
                   chunk_text: str) -> Dict[str, Any]:
    """
    Returns validation result with:
        in_bounds, strict_match, token_similarity, note
    """
    if char_idx == [-1, -1]:
        # Span legitimately comes from an evidence chunk
        return {
            "in_bounds": True,
            "strict_match": False,
            "token_similarity": 0.0,
            "note": "evidence_chunk_span"
        }

    length = len(chunk_text)
    start, end = clip_span(char_idx[0], char_idx[1], length)

    if start == -1 or end == -1:
        return {
            "in_bounds": False,
            "strict_match": False,
            "token_similarity": 0.0,
            "note": "out_of_bounds"
        }

    extracted = chunk_text[start:end]
    strict = (extracted == item_span)
    token_sim = token_jaccard(extracted, item_span)

    return {
        "in_bounds": True,
        "strict_match": strict,
        "token_similarity": round(token_sim, 4),
        "note": "" if strict else "token_fallback"
    }


# --------------------------------------------------------------------------- #
# Main validation
# --------------------------------------------------------------------------- #
def main() -> None:
    chunk_text_map = load_chunk_text()

    factors = load_jsonl(EXTRACTION_DIR / "factors.jsonl")
    relations = load_jsonl(EXTRACTION_DIR / "relations.jsonl")
    attributions = load_jsonl(EXTRACTION_DIR / "attributions.jsonl")

    high_factors, review_factors = [], []
    high_relations, review_relations = [], []
    high_attribs, review_attribs = [], []

    stats = {
        "factors": {"total": len(factors), "high_conf": 0, "low_conf": 0, "span_mismatch": 0},
        "relations": {"total": len(relations), "high_conf": 0, "low_conf": 0, "span_mismatch": 0},
        "attributions": {"total": len(attributions), "high_conf": 0, "low_conf": 0}
    }

    # ---- Factors ----
    for f in factors:
        chunk_uid = f.get("chunk_uid", "")
        text = chunk_text_map.get(chunk_uid, "")

        validation = validate_span(
            item_span=f.get("text_span", ""),
            char_idx=f.get("char_idx", [-1, -1]),
            chunk_text=text
        )
        f["_validation"] = validation

        low_conf = f.get("confidence", 0.0) < CONFIDENCE_THRESHOLD_FACTOR
        mismatch = (not validation["in_bounds"] or
                    (not validation["strict_match"] and validation["token_similarity"] < TOKEN_SIMILARITY_THRESHOLD))

        if low_conf or mismatch:
            review_factors.append(f)
            if low_conf:
                stats["factors"]["low_conf"] += 1
            if mismatch:
                stats["factors"]["span_mismatch"] += 1
        else:
            high_factors.append(f)
            stats["factors"]["high_conf"] += 1

    # ---- Relations ----
    for r in relations:
        chunk_uid = r.get("chunk_uid", "")
        text = chunk_text_map.get(chunk_uid, "")

        validation = validate_span(
            item_span=r.get("evidence_span", ""),
            char_idx=r.get("char_idx", [-1, -1]),
            chunk_text=text
        )
        r["_validation"] = validation

        low_conf = r.get("confidence", 0.0) < CONFIDENCE_THRESHOLD_RELATION
        mismatch = (not validation["in_bounds"] or
                    (not validation["strict_match"] and validation["token_similarity"] < TOKEN_SIMILARITY_THRESHOLD))

        if low_conf or mismatch:
            review_relations.append(r)
            if low_conf:
                stats["relations"]["low_conf"] += 1
            if mismatch:
                stats["relations"]["span_mismatch"] += 1
        else:
            high_relations.append(r)
            stats["relations"]["high_conf"] += 1

    # ---- Attributions (only confidence threshold) ----
    for a in attributions:
        low_conf = a.get("confidence", 0.0) < CONFIDENCE_THRESHOLD_ATTRIB
        if low_conf:
            review_attribs.append(a)
            stats["attributions"]["low_conf"] += 1
        else:
            high_attribs.append(a)
            stats["attributions"]["high_conf"] = stats["attributions"].get("high_conf", 0) + 1

    # ---- Write outputs ----
    write_jsonl(HIGH_CONF_DIR / "factors.highconf.jsonl", high_factors)
    write_jsonl(HIGH_CONF_DIR / "relations.highconf.jsonl", high_relations)
    write_jsonl(HIGH_CONF_DIR / "attributions.highconf.jsonl", high_attribs)

    write_jsonl(REVIEW_DIR / "factors.review.jsonl", review_factors)
    write_jsonl(REVIEW_DIR / "relations.review.jsonl", review_relations)
    write_jsonl(REVIEW_DIR / "attributions.review.jsonl", review_attribs)

    # ---- Report ----
    report = {
        "timestamp_utc": Path(__file__).stat().st_mtime,
        "thresholds": {
            "confidence_factor": CONFIDENCE_THRESHOLD_FACTOR,
            "confidence_relation": CONFIDENCE_THRESHOLD_RELATION,
            "confidence_attribution": CONFIDENCE_THRESHOLD_ATTRIB,
            "token_similarity": TOKEN_SIMILARITY_THRESHOLD
        },
        "summary": stats,
        "outputs": {
            "high_confidence_dir": str(HIGH_CONF_DIR),
            "review_queue_dir": str(REVIEW_DIR)
        }
    }

    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Evidence validation complete.")
    print(f"  High-confidence factors     : {len(high_factors)} / {len(factors)}")
    print(f"  High-confidence relations  : {len(high_relations)} / {len(relations)}")
    print(f"  High-confidence attributions: {len(high_attribs)} / {len(attributions)}")
    print(f"  Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()