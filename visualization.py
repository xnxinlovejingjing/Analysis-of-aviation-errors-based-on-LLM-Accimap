import json
import re
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# --------------------------------------------------------------------------- #
# Paths & output
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).parent.resolve()
EXTRACTION_DIR = BASE_DIR / "extraction_results"
HIGH_CONF_DIR = EXTRACTION_DIR / "high_confidence"
FIGURE_DIR = EXTRACTION_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Nature-style matplotlib settings
mpl.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Muted, colour-blind friendly palette
PALETTE = {
    "factors": "#1f77b4",      # muted blue
    "relations": "#ff7f0e",    # muted orange
    "attributions": "#2ca02c", # muted green
    "design": "#d62728",       # red
    "execution": "#9467bd",    # purple
    "mixed": "#8c564b",        # brown
    "uncertain": "#7f7f7f",    # grey
}

# --------------------------------------------------------------------------- #
# Load data
# --------------------------------------------------------------------------- #
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

factors_all = load_jsonl(EXTRACTION_DIR / "factors.jsonl")
relations_all = load_jsonl(EXTRACTION_DIR / "relations.jsonl")
attributions_all = load_jsonl(EXTRACTION_DIR / "attributions.jsonl")

factors_high = load_jsonl(HIGH_CONF_DIR / "factors.highconf.jsonl")
relations_high = load_jsonl(HIGH_CONF_DIR / "relations.highconf.jsonl")
attributions_high = load_jsonl(HIGH_CONF_DIR / "attributions.highconf.jsonl")

# --------------------------------------------------------------------------- #
# Helper
# --------------------------------------------------------------------------- #
def save_fig(fig: plt.Figure, name: str) -> None:
    for ext in ["png", "pdf"]:
        fig.savefig(FIGURE_DIR / f"{name}.{ext}", bbox_inches="tight")
    print(f"Saved {name}.png & {name}.pdf")

# --------------------------------------------------------------------------- #
# 1. Confidence distributions
# --------------------------------------------------------------------------- #
fig, axs = plt.subplots(1, 3, figsize=(7.5, 2.5))

for data, label, ax, color in zip(
    [factors_all, relations_all, attributions_all],
    ["Factors", "Relations", "Attributions"],
    axs, [PALETTE["factors"], PALETTE["relations"], PALETTE["attributions"]]
):
    confs = [item.get("confidence", 0.0) for item in data]
    ax.hist(confs, bins=20, range=(0, 1), color=color, alpha=0.8, edgecolor="black", linewidth=0.4)
    ax.set_title(label)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.grid(True, ls="--", lw=0.4, alpha=0.6)

plt.tight_layout()
save_fig(fig, "fig_confidence_distributions")
plt.close(fig)

# --------------------------------------------------------------------------- #
# 2. Token similarity sensitivity (auto-accept ratio)
# --------------------------------------------------------------------------- #
thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
results = []

chunk_text = {obj["id"]: obj["text"] for obj in load_jsonl(BASE_DIR / "knowledge_base" / "kb_meta.jsonl")}

for th in thresholds:
    factor_ok = 0
    rel_ok = 0
    for f in factors_all:
        txt = chunk_text.get(f.get("chunk_uid", ""), "")
        span = f.get("text_span", "")
        idx = f.get("char_idx", [-1, -1])
        if idx == [-1, -1]:
            factor_ok += 1
            continue
        s, e = max(0, idx[0]), max(0, idx[1])
        extracted = txt[s:e] if s < e and e <= len(txt) else ""
        if extracted == span or (token_jaccard := len(set(re.findall(r"\w+", extracted.lower())) & set(re.findall(r"\w+", span.lower()))) / max(1, len(set(re.findall(r"\w+", extracted.lower())) | set(re.findall(r"\w+", span.lower()))))) >= th:
            factor_ok += 1
    for r in relations_all:
        txt = chunk_text.get(r.get("chunk_uid", ""), "")
        span = r.get("evidence_span", "")
        idx = r.get("char_idx", [-1, -1])
        if idx == [-1, -1]:
            rel_ok += 1
            continue
        s, e = max(0, idx[0]), max(0, idx[1])
        extracted = txt[s:e] if s < e and e <= len(txt) else ""
        if extracted == span or (len(set(re.findall(r"\w+", extracted.lower())) & set(re.findall(r"\w+", span.lower()))) / max(1, len(set(re.findall(r"\w+", extracted.lower())) | set(re.findall(r"\w+", span.lower()))))) >= th:
            rel_ok += 1

    results.append({
        "threshold": th,
        "factors_ratio": factor_ok / max(1, len(factors_all)),
        "relations_ratio": rel_ok / max(1, len(relations_all))
    })

df_sens = pd.DataFrame(results)

fig, ax = plt.subplots(figsize=(3.8, 2.8))
x = np.arange(len(thresholds))
width = 0.35
ax.bar(x - width/2, df_sens["factors_ratio"]*100, width, label="Factors", color=PALETTE["factors"], edgecolor="black", linewidth=0.5)
ax.bar(x + width/2, df_sens["relations_ratio"]*100, width, label="Relations", color=PALETTE["relations"], edgecolor="black", linewidth=0.5)
ax.set_xlabel("Token similarity threshold")
ax.set_ylabel("Auto-accept ratio (%)")
ax.set_xticks(x)
ax.set_xticklabels([f"{t:.2f}" for t in thresholds])
ax.set_ylim(0, 105)
ax.grid(True, axis="y", ls="--", lw=0.4, alpha=0.7)
ax.legend(frameon=False)
for i, row in df_sens.iterrows():
    ax.text(i - width/2, row["factors_ratio"]*100 + 2, f"{row['factors_ratio']*100:.1f}", ha="center", va="bottom", fontsize=7)
    ax.text(i + width/2, row["relations_ratio"]*100 + 2, f"{row['relations_ratio']*100:.1f}", ha="center", va="bottom", fontsize=7)
plt.tight_layout()
save_fig(fig, "fig_token_similarity_sensitivity")
plt.close(fig)

# --------------------------------------------------------------------------- #
# 3. Accimap level & type distribution (high-confidence only)
# --------------------------------------------------------------------------- #
level_counts = pd.Series([f["level"] for f in factors_high]).value_counts()
type_counts = pd.Series([f["type"] for f in factors_high]).value_counts()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))
ax1.bar(level_counts.index, level_counts.values, color=PALETTE["factors"], edgecolor="black", linewidth=0.5)
ax1.set_title("Accimap Level Distribution")
ax1.set_ylabel("Count")
ax1.tick_params(axis="x", rotation=45)

ax2.bar(type_counts.index, type_counts.values, color=PALETTE["relations"], edgecolor="black", linewidth=0.5)
ax2.set_title("Factor Type Distribution")
ax2.tick_params(axis="x", rotation=45)

plt.tight_layout()
save_fig(fig, "fig_level_type_distribution")
plt.close(fig)

# --------------------------------------------------------------------------- #
# 4. Design vs Execution attribution
# --------------------------------------------------------------------------- #
axis_counts = pd.Series([a["axis"] for a in attributions_high]).value_counts()
labels = ["Design flaw", "Execution gap", "Mixed", "Uncertain"]
colors = [PALETTE["design"], PALETTE["execution"], PALETTE["mixed"], PALETTE["uncertain"]]

fig, ax = plt.subplots(figsize=(3.5, 3.0))
wedges, texts, autotexts = ax.pie(
    axis_counts.reindex(["design_flaw", "execution_gap", "mixed", "uncertain"], fill_value=0),
    labels=labels,
    colors=colors,
    autopct="%1.1f%%",
    startangle=90
)
for autotext in autotexts:
    autotext.set_color("white")
    autotext.set_fontweight("bold")
ax.set_title("Design vs Execution Attribution")
save_fig(fig, "fig_attribution_pie")
plt.close(fig)

print(f"All figures saved to {FIGURE_DIR.resolve()}")