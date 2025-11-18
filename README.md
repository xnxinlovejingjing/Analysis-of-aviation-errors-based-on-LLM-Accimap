# Analysis of aviation errors based on LLM-Accimap

Official code repository accompanying the paper  
**"Systemic Risk Analysis of Aviation Safety Using an LLM–Accimap Framework"**

This pipeline performs fully reproducible, evidence-bound extraction of Accimap factors, causal relations, and design-vs-execution attribution from English-language regulatory documents and incident reports using a retrieval-augmented large language model (Qwen-Plus via DashScope).

Key features:
- Strict character-level evidence binding with token-similarity fallback
- Confidence thresholding + human-in-the-loop review queue
- Balanced same-side / cross-side retrieval
- No external knowledge or hallucination allowed
- Publication-ready visualisation scripts

## Quick start

```bash
# 1. Clone & install
git clone https://github.com/yourname/aviation-accimap-llm.git
cd aviation-accimap-llm
pip install -r requirements.txt

# 2. Download spaCy model
python -m spacy download en_core_web_sm

# 3. (Optional) Build knowledge base with local embeddings
python build_knowledge_base.py

#    Or with DashScope:
python build_knowledge_base.py --use-dashscope --api-key sk-...

# 4. Run extraction (requires DashScope API key)
python llm_guided_extraction.py --api-key sk-...

# 5. Validate & filter high-confidence results
python evidence_validation.py

# 6. Generate figures
python visualization.py

