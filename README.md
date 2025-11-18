# Analysis of aviation errors based on LLM-Accimap

Official code repository accompanying the paper  
**"Schema-Guided LLM Extraction with Evidence Binding for Accimap-Based Failure Attribution in Aviation Emergency Response"**

This pipeline performs fully reproducible, evidence-bound extraction of Accimap factors, causal relations, and design-vs-execution attribution from English-language regulatory documents and incident reports using a retrieval-augmented large language model (Qwen-Plus via DashScope).

Key features:
- Strict character-level evidence binding with token-similarity fallback
- Confidence thresholding + human-in-the-loop review queue
- Balanced same-side / cross-side retrieval
- No external knowledge or hallucination allowed
- Publication-ready visualisation scripts

## Repository structure
