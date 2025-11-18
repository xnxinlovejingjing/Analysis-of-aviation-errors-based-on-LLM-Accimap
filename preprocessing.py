import os
import re
import json
import hashlib
import random
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import contractions
import nltk
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
from gensim.models import Word2Vec

# --------------------------------------------------------------------------- #
# Configuration (adjust to your environment)
# --------------------------------------------------------------------------- #
INPUT_FILES = [
    # Add your regulatory / interview texts here
    # Example (publicly available NTSB-style reports can be used for demonstration):
    # "data/regulations_part1.txt",
    # "data/regulations_part2.txt",
    # "data/interview_transcripts.txt",
]

OUTPUT_DIR = Path("preprocessed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20250918
random.seed(SEED)
np.random.seed(SEED)

# --------------------------------------------------------------------------- #
# NLP resources
# --------------------------------------------------------------------------- #
nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

try:
    EN_STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    EN_STOPWORDS = set(stopwords.words("english"))

DOMAIN_KEEP_TERMS = {
    "icao", "faa", "caac", "ntsb", "jtsb", "easa",
    "icao-annex", "icao_doc", "ccar"
}

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def normalize_text(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("—", "-").replace("–", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_document(raw_text: str) -> Dict[str, Any]:
    raw_text = normalize_text(raw_text)
    raw_text = contractions.fix(raw_text)
    raw_text = re.sub(r"https?://\S+|www\.\S+|\S+@\S+", " ", raw_text)

    doc = nlp(raw_text)

    tokens: List[str] = []
    sent_tokens: List[List[str]] = []

    for sent in doc.sents:
        sent_toks = []
        for token in sent:
            if not token.is_alpha:
                continue
            lemma = token.lemma_.lower()
            if len(lemma) < 2:
                continue
            if lemma in EN_STOPWORDS and lemma not in DOMAIN_KEEP_TERMS:
                continue
            sent_toks.append(lemma)
        if sent_toks:
            sent_tokens.append(sent_toks)
            tokens.extend(sent_toks)

    cleaned = " ".join(tokens)
    return {
        "cleaned_text": cleaned,
        "tokens": tokens,
        "sent_tokens": sent_tokens
    }


def document_id_from_path(path: str) -> str:
    hash_val = hashlib.md5(path.encode("utf-8")).hexdigest()[:8]
    return f"doc_{hash_val}"


# --------------------------------------------------------------------------- #
# Main processing
# --------------------------------------------------------------------------- #
def main() -> None:
    documents_meta: List[Dict[str, Any]] = []
    cleaned_records: List[Dict[str, Any]] = []
    token_records: List[Dict[str, Any]] = []
    tfidf_corpus: List[str] = []

    for filepath in INPUT_FILES:
        if not Path(filepath).exists():
            print(f"[Warning] File not found, skipping: {filepath}")
            continue

        raw_text = Path(filepath).read_text(encoding="utf-8", errors="ignore")
        doc_id = document_id_from_path(filepath)
        cleaned = clean_document(raw_text)

        # Per-document clean text (useful for inspection)
        (OUTPUT_DIR / f"{doc_id}.clean.txt").write_text(cleaned["cleaned_text"], encoding="utf-8")

        documents_meta.append({
            "doc_id": doc_id,
            "source_path": filepath,
            "token_count": len(cleaned["tokens"])
        })
        cleaned_records.append({"doc_id": doc_id, "cleaned_text": cleaned["cleaned_text"]})
        token_records.append({"doc_id": doc_id, "tokens": cleaned["tokens"]})
        tfidf_corpus.append(cleaned["cleaned_text"])

    # JSONL dumps
    with open(OUTPUT_DIR / "cleaned_docs.jsonl", "w", encoding="utf-8") as f:
        for rec in cleaned_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(OUTPUT_DIR / "docs_tokens.jsonl", "w", encoding="utf-8") as f:
        for rec in token_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(OUTPUT_DIR / "doc_index.json", "w", encoding="utf-8") as f:
        json.dump(documents_meta, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------- #
    # TF-IDF
    # ------------------------------------------------------------------- #
    min_df = 1 if len(tfidf_corpus) <= 5 else 2
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=0.90,
        lowercase=False,
        sublinear_tf=True,
        smooth_idf=True,
        dtype=np.float32
    )
    tfidf_matrix = vectorizer.fit_transform(tfidf_corpus)
    save_npz(OUTPUT_DIR / "tfidf.npz", tfidf_matrix)

    vocab = vectorizer.get_feature_names_out().tolist()
    with open(OUTPUT_DIR / "tfidf_vocabulary.json", "w", encoding="utf-8") as f:
        json.dump({"vocabulary": vocab}, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------- #
    # Word2Vec (sentence chunks for training stability)
    # ------------------------------------------------------------------- #
    sentences: List[List[str]] = []
    for rec in token_records:
        tokens = rec["tokens"]
        chunk_size = 30
        for i in range(0, len(tokens), chunk_size):
            sentences.append(tokens[i:i + chunk_size])

    w2v = Word2Vec(
        sentences=sentences,
        vector_size=200,
        window=5,
        min_count=2,
        workers=os.cpu_count() or 4,
        sg=1,
        negative=10,
        epochs=20,
        seed=SEED
    )
    w2v.save(str(OUTPUT_DIR / "word2vec.model"))
    w2v.wv.save(str(OUTPUT_DIR / "word2vec.keyedvectors.kv"))

    print(f"Preprocessing complete. Outputs written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()