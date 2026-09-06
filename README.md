# Vietnamese LegalQA — Hierarchical Hybrid Retrieval

A competition-oriented pipeline for Vietnamese Legal Question Answering using:

- **Approach A — Hierarchical metadata enrichment**
- **Approach B — Multi-granularity parent/child retrieval**
- **Approach E — Hierarchy expansion / sibling rescue**
- **SQLite FTS5 BM25 + Vietnamese dense embeddings**
- **Reciprocal Rank Fusion (RRF)**
- **Vietnamese reranker + threshold tuning**
- **Strict separation between retrieval context and generation evidence**
- **Optional GenAI for one short subject-aware intro sentence only**

The legal evidence body is always appended **verbatim from the selected context corpus** and is never rewritten by the LLM.

---

## 1. Architecture

```text
selected-contexts
        |
        v
Flexible Legal Parser
        |
        v
Hierarchical Legal Tree
        |
        +-------------------------------+
        |                               |
        v                               v
retrieval_text                       raw_text
(parent headings + own text)        (own legal evidence)
        |                               |
        |                               +-----------------------+
        |                                                       |
        v                                                       v
SQLite FTS5 BM25                  GENERATION BOUNDARY       evidence body
        |                                                       ^
        |             +----------------+                        |
        +------------>|      RRF       |                        |
                      +-------+--------+                        |
        +-------------------->|                                 |
        |                     v                                 |
        |                  Top-K                                |
        |                     |                                 |
        v                     v                                 |
Vietnamese Dense       Hierarchy Expansion                      |
Embedding + FAISS      parent / child / siblings                |
                              |                                 |
                              v                                 |
                       Vietnamese Reranker                      |
                              |                                 |
                              v                                 |
                         threshold tau                           |
                              |                                 |
                              v                                 |
                       Evidence Compaction                       |
                              |                                 |
                              v                                 |
                       Original legal order                     |
                              |                                 |
                              +----------> intro only ----------+
                                           optional LLM
```

### Core rule

The project stores two distinct representations for every legal node:

**Retrieval representation** may contain ancestor headings such as document name, Chapter, Section, Article and Clause titles.

**Generation representation** is the node's original `raw_text` only.

Therefore hierarchy can improve retrieval without leaking parent body text into the final legal evidence.

---

## 2. Project structure

```text
legalqa-hierarchical-rag/
├── configs/
│   └── default.yaml
├── data/
│   └── .gitkeep
├── runs/
│   └── .gitkeep
├── scripts/
│   ├── build_index.py
│   ├── infer.py
│   ├── inspect_parser.py
│   └── tune_threshold.py
├── src/legalqa/
│   ├── config.py
│   ├── database.py
│   ├── dense.py
│   ├── evaluation.py
│   ├── evidence.py
│   ├── fusion.py
│   ├── generation.py
│   ├── hierarchy.py
│   ├── io.py
│   ├── parser.py
│   ├── pipeline.py
│   ├── reranker.py
│   ├── schema.py
│   ├── sparse.py
│   └── utils.py
├── tests/
│   └── test_parser.py
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 3. Data placement

Put the competition files in `data/`:

```text
data/
├── selected-contexts.zip
├── train.json
└── public-official.json
```

`selected-contexts.zip` can also be extracted into a directory. Set the correct path in `configs/default.yaml`.

The `.gitignore` excludes datasets, SQLite files, FAISS indexes, model caches and experiment outputs so they are not accidentally pushed to GitHub.

---

## 4. Environment setup

Recommended Python: **3.10+**.

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Check GPU:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The embedding model and reranker automatically use CUDA when `runtime.device: auto` and CUDA is available.

FAISS in `requirements.txt` uses `faiss-cpu`. This is sufficient for the exact `IndexFlatIP` baseline. Embedding generation and reranking are usually the heavier GPU workloads.

---

## 5. Configure models

Edit:

```text
configs/default.yaml
```

Set:

```yaml
models:
  embedding_model: YOUR_HUGGINGFACE_VIETNAMESE_EMBEDDING_MODEL
  reranker_model: YOUR_HUGGINGFACE_VIETNAMESE_RERANKER
  intro_llm_model: ""
```

The project intentionally does **not** hard-code a model repository because the exact Vietnamese embedding/reranker should be treated as an experimental variable.

### Reranker compatibility

`src/legalqa/reranker.py` assumes a Hugging Face sequence-classification reranker:

```python
AutoModelForSequenceClassification
```

If your selected reranker has a custom `compute_score()` API, only adapt `VietnameseReranker.score()`. The retrieval architecture does not need to change.

---

## 6. Inspect parser before indexing

The corpus mixes legal documents, decisions, technical standards and procedures. Inspect several parsed trees first:

```bash
cd scripts
python .\inspect_parser.py --source ..\data\selected-contexts.zip --limit 3 --max-nodes 80
```

The flexible parser detects structures such as:

```text
PHẦN -> CHƯƠNG -> MỤC -> ĐIỀU -> KHOẢN -> ĐIỂM
```

and technical numbering:

```text
3
3.2
3.2.1
3.2.1.1
```

as well as forms such as:

```text
I.
1.
a)
```

Node IDs use source offsets rather than legal labels, so repeated `Điều 1` blocks inside the same passage remain unique.

---

## 7. Build SQLite BM25 + FAISS

After configuring the embedding model:

```bash
python scripts/build_index.py --config configs/default.yaml
```

Outputs:

```text
artifacts/legal_nodes.sqlite
artifacts/legal_dense.faiss
runs/default/index_stats.json
```

### SQLite stores the full tree

All detected structural nodes are stored in `nodes` so parent/child relations remain available.

Only retrieval-relevant node types are indexed in FTS/FAISS, including:

```text
article
clause
point
numeric_section
roman_section
numbered_item
letter_item
```

---

## 8. Retrieval pipeline

For each question:

### Step 1 — Sparse

```text
SQLite FTS5 BM25 -> Top 100
```

BM25 searches two fields:

```text
context_text
raw_text
```

`context_text` contains legal ancestor headings, while `raw_text` contains the original node text.

### Step 2 — Dense

```text
Vietnamese embedding -> FAISS cosine/IP -> Top 100
```

Dense vectors are built from:

```text
retrieval_text = context_text + raw_text
```

### Step 3 — RRF

Sparse and dense ranks are fused with Reciprocal Rank Fusion:

```text
score(c) = ws/(k + rank_sparse) + wd/(k + rank_dense)
```

Default:

```yaml
rrf_k: 60
sparse_weight: 1.0
dense_weight: 1.0
fusion_top_k: 50
```

RRF is used instead of directly adding BM25 and cosine scores because the score scales are different.

### Step 4 — Hierarchy expansion

Strong Top-K seeds are expanded with:

```text
parent
children
siblings
one additional upward sibling rescue
```

This is designed for cases where one provision contains the penalty while nearby provisions contain additional sanctions or remedies.

### Step 5 — Reranker

The reranker receives:

```text
(question, retrieval_text)
```

and assigns relevance scores to the expanded candidate set.

### Step 6 — Threshold

Only candidates above `tau` are retained, with a fallback to the highest-scoring candidate when no node passes the threshold.

### Step 7 — Evidence compaction

The system removes redundant parent/child evidence and attempts to select the **smallest sufficient legal unit**.

If a parent and its relevant child nodes both pass the threshold, only the
fine-grained child nodes are retained. Relevant children are never promoted back
to the broader parent because the parent's text may contain unrelated children.

Finally, evidence is sorted by original source offsets rather than reranker score.

---

## 9. Single-question inference

```bash
python scripts/infer.py \
  --config configs/default.yaml \
  --question "Nguồn nhân lực cho công tác phòng chống thiên tai gồm những gì?" \
  --debug-jsonl runs/default/debug.jsonl
```

Override threshold temporarily:

```bash
python scripts/infer.py \
  --config configs/default.yaml \
  --question "..." \
  --threshold 0.60
```

---

## 10. Public test inference

On Windows PowerShell, the simplest resumable command is:

```powershell
.\create_submission.ps1
```

It writes `data/submission.json`, checkpoints every 10 new answers, and resumes
from that file after an interruption. Use `-Fresh` to intentionally recompute all
answers, or override candidate count with `-SparseTopK 50`.

```bash
python scripts/infer.py \
  --config configs/default.yaml \
  --input data/public-official.json \
  --output runs/default/public_predictions.json \
  --debug-jsonl runs/default/public_debug.jsonl
```

Output format:

```json
{
  "80189": {
    "question": "...",
    "answer": "..."
  }
}
```

Adjust the final submission serializer if the competition platform expects a different exact schema.

---

## 11. Answer formatting

The final answer keeps one introductory sentence followed by bulleted evidence.
Lines are joined with the Python newline character `"\n"`. Source line wrapping
is collapsed into spaces, while legal list markers such as `1.`, `2)` and `a)`
are replaced by one consistent `- ` prefix:

```text
Đối với chủ đề được hỏi, các quy định liên quan như sau:
- Nội dung thứ nhất được trình bày trên một dòng hoàn chỉnh.
- Nội dung thứ hai.
```

No numeric list markers are inserted before the evidence items.

---

## 12. Threshold tuning

`public-official.json` has no gold answers, so threshold tuning should use a validation split from `train.json`.

Quick experiment:

```bash
python scripts/tune_threshold.py \
  --config configs/default.yaml \
  --val-ratio 0.15 \
  --limit 100
```

Full sweep:

```bash
python scripts/tune_threshold.py \
  --config configs/default.yaml \
  --thresholds 0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75
```

The script caches reranker outputs once:

```text
runs/default/rerank_cache.json
```

Therefore threshold sweeps do not need to rerun dense retrieval and reranking every time.

Results:

```text
runs/default/threshold_results.json
```

### Important evaluation note

The included METEOR implementation is an **approximate local proxy** based on tokenized Vietnamese text. If the competition organizers release an official scorer/tokenizer, replace the local scorer with that implementation before making final decisions.

---

## 13. Recommended first experiments

Run these before adding more complexity:

| Run | Retrieval | Hierarchy expansion | Intro |
|---|---|---|---|
| R1 | A + B | No | deterministic |
| R2 | A + B | **Yes (E)** | deterministic |
| R3 | A + B | **Yes (E)** | GenAI intro only |

The key comparison is **R2 vs R1**. If R2 improves validation METEOR/ROUGE, sibling rescue is providing useful evidence recall.

Then compare **R3 vs R2** to determine whether GenAI improves metric score or merely makes the response more readable.

---

## 14. Important hyperparameters

Start with:

```yaml
retrieval:
  sparse_top_k: 100
  dense_top_k: 100
  rrf_k: 60
  sparse_weight: 1.0
  dense_weight: 1.0
  fusion_top_k: 50

hierarchy:
  expansion_seed_k: 20
  max_neighbors_per_seed: 24
  max_expanded: 150

reranker:
  threshold: 0.55

answer:
  max_evidence_nodes: 6
```

Tune in roughly this order:

1. reranker threshold;
2. fusion Top-K;
3. hierarchy expansion size/depth;
4. max evidence nodes;
5. sparse/dense RRF weights;
6. embedding model;
7. reranker model.

---

## 15. Debugging retrieval

Every inference can write a JSONL record containing:

```text
question
sparse_top
dense_top
fused_top
expanded_count
reranked_top
threshold
selected evidence
final answer
```

Example:

```bash
python scripts/infer.py \
  --config configs/default.yaml \
  --question "..." \
  --debug-jsonl runs/debug.jsonl
```

This makes it possible to determine whether a failure came from:

```text
parser
-> sparse/dense recall
-> RRF
-> hierarchy expansion
-> reranker
-> threshold
-> evidence compaction
-> intro generation
```

rather than treating the system as one opaque RAG pipeline.

---

## 16. Run tests

```bash
pytest -q
```

Current tests verify basic Article/Clause/Point parsing and unique IDs when an Article number restarts inside one passage.

---

## 17. Git usage

Initialize a repository:

```bash
git init
git add .
git commit -m "Initial hierarchical LegalQA pipeline"
```

Then connect your remote repository:

```bash
git branch -M main
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

Competition data, indexes and experiment logs are excluded by `.gitignore`.

---

## 18. Current design decisions

### Why not prepend parent body to child evidence?

Because parent text is useful for retrieval but may pollute the final output. The project enriches `retrieval_text` with ancestor headings while keeping `raw_text` clean.

### Why multiple granularities?

Article-level chunks provide context; Clause/Point chunks provide precision. Searching both reduces the trade-off between context and specificity.

### Why expansion before reranking?

A relevant sibling may rank outside initial Top-50. Expanding strong legal neighborhoods first gives the reranker a chance to rescue it.

### Why RRF?

BM25 and cosine scores are not directly comparable. Rank fusion avoids score-scale calibration in the initial baseline.

### Why deterministic intro by default?

It removes generation variance while retrieval is being tuned. GenAI can be enabled later as a separate ablation.

---

## 19. Suggested next improvements

After the baseline is stable, useful experiments include:

- domain-specific Vietnamese text normalization for BM25;
- query expansion using legal synonyms without altering the answer;
- separate dense vectors for raw text vs hierarchy context;
- learned sparse/dense fusion;
- document-aware reranker batching;
- better rule-based structural parsing for specific document families;
- answer-style calibration using train data without using train answers as legal evidence;
- official METEOR scorer integration if released.

Keep these as separate experiments so the effect of hierarchical retrieval remains measurable.

<<<<<<< HEAD
## 20. Result in codabench 0.4016
=======

## 20. Inference public-official
```bash
python infer.py --config configs/default.yaml --input data/public-official.json --output data/submission.json --resume --checkpoint-every 10
```
>>>>>>> cf22951 (Sparse + Reranker 0.43 with countNode and point_clause_length code)
