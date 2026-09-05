# Vietnamese LegalQA — Hierarchical Sparse Retrieval

A competition-oriented pipeline for Vietnamese Legal Question Answering using:

- **Approach A — Hierarchical metadata enrichment**
- **Approach B — Multi-granularity parent/child retrieval**
- **SQLite FTS5 BM25 sparse retrieval**
- **Direct BM25 Top-K → Vietnamese reranker flow**
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
        v                               |
SQLite FTS5 BM25                       |
        |                               |
        v                               |
Top-K sparse candidates                |
        |                               |
        v                               |
Vietnamese Reranker                    |
        |                               |
        v                               |
threshold tau -> Evidence Compaction <-+
        |
        v
Original legal order -> optional intro -> final answer
```

### Core rule

The project stores two distinct representations for every legal node:

**Retrieval representation** may contain ancestor headings such as document name, Chapter, Section, Article and Clause titles.

**Generation representation** is the node's original `raw_text` only.

Therefore hierarchical metadata can improve sparse retrieval without leaking parent body text into the final legal evidence.

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
│   ├── evaluation.py
│   ├── evidence.py
│   ├── generation.py
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

The `.gitignore` excludes datasets, SQLite files, model caches and experiment outputs so they are not accidentally pushed to GitHub.

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

The reranker automatically uses CUDA when `runtime.device: auto` and CUDA is available. BM25 runs in SQLite and does not use an embedding model or FAISS.

The configured XLM-R reranker tokenizer requires `sentencepiece` and `protobuf`.
They are included in `requirements.txt`; after dependency changes, run:

```bash
python -m pip install -r requirements.txt
```

When the reranker is already present in the Hugging Face cache, the runtime uses
that local snapshot directly. This avoids an unnecessary Hub metadata request
and is useful on Windows machines whose Python certificate store cannot verify
the local network's HTTPS certificate chain.

---

## 5. Configure models

Edit:

```text
configs/default.yaml
```

Set:

```yaml
models:
  reranker_model: YOUR_HUGGINGFACE_VIETNAMESE_RERANKER
  intro_llm_model: ""
```

The reranker remains an experimental variable and can be changed independently of sparse retrieval.

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
python scripts/inspect_parser.py \
  --source data/selected-contexts.zip \
  --limit 5
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

## 7. Build the SQLite BM25 index

```bash
python scripts/build_index.py --config configs/default.yaml
```

Outputs:

```text
artifacts/legal_nodes.sqlite
runs/default/index_stats.json
```

### SQLite stores the full tree

All detected structural nodes are stored in `nodes` so parent/child relations remain available.

Only retrieval-relevant node types are indexed in FTS5, including:

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

For each question, the runtime path is deliberately simple:

### Step 1 — Sparse candidate filtering

```text
SQLite FTS5 BM25 -> Top-K
```

BM25 searches two fields:

```text
context_text
raw_text
```

`context_text` contains legal ancestor headings, while `raw_text` contains the original node text. The FTS table searches both fields. No embedding model, dense index, RRF, or hierarchy expansion is executed.

### Step 2 — Reranker

The reranker receives:

```text
(question, retrieval_text)
```

and assigns relevance scores to exactly the BM25 candidate set. If BM25 returns fewer than Top-K matches, the reranker receives only those matches.

The candidate-set size is controlled in `configs/default.yaml`:

```yaml
retrieval:
  sparse_top_k: 100
```

For example, set it to `50` to feed at most 50 candidates, or `200` to feed at most 200 candidates. This parameter affects retrieval recall and reranking time.

Do not confuse it with:

```yaml
reranker:
  batch_size: 32
```

`batch_size` only controls how many pairs are scored in one GPU/CPU mini-batch; it does not reduce the total candidate set.

### Step 3 — Threshold

Only candidates above `tau` are retained, with a fallback to the highest-scoring candidate when no node passes the threshold.

### Step 4 — Evidence compaction

The system removes redundant parent/child evidence and attempts to select the **smallest sufficient legal unit**.

If multiple relevant Points under the same Clause are selected and the parent Clause is also sufficiently relevant, the system can promote them to the Clause to avoid fragmented evidence.

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

Override the BM25/reranker candidate-set size temporarily, without editing YAML:

```bash
python scripts/infer.py \
  --config configs/default.yaml \
  --question "..." \
  --sparse-top-k 50
```

---

## 10. Public test inference

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

## 11. Controlled answer generation

Default configuration:

```yaml
answer:
  use_llm_intro: false
```

This uses a deterministic subject-aware intro such as:

```text
Đối với nguồn nhân lực cho công tác phòng chống thiên tai,
các quy định liên quan tại Điều 6 như sau:
```

Then the program appends retrieved `raw_text` verbatim.

### Optional GenAI intro

Configure:

```yaml
models:
  intro_llm_model: YOUR_GENERATIVE_MODEL

answer:
  use_llm_intro: true
```

The LLM receives only:

- the question;
- a derived subject hint;
- the selected legal paths.

It is instructed to produce exactly one intro sentence and not to create legal content.

**The evidence body never passes through the LLM.**

This boundary is intentional for METEOR/ROUGE-oriented competition evaluation and hallucination control.

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

Therefore threshold sweeps do not need to rerun sparse retrieval and reranking every time. Cache metadata includes `sparse_top_k` and the reranker settings, so an incompatible cache is ignored automatically.

Results:

```text
runs/default/threshold_results.json
```

### Important evaluation note

The included METEOR implementation is an **approximate local proxy** based on tokenized Vietnamese text. If the competition organizers release an official scorer/tokenizer, replace the local scorer with that implementation before making final decisions.

---

## 13. Recommended candidate-set experiment

Try the same validation subset with values such as `25, 50, 100, 200`. Larger values can improve BM25 recall, but reranking becomes slower roughly in proportion to the number of candidates. Choose the smallest value that preserves validation quality.

Example:

```bash
python scripts/tune_threshold.py \
  --config configs/default.yaml \
  --limit 100 \
  --sparse-top-k 50 \
  --cache runs/default/rerank_cache_k50.json
```

---

## 14. Important hyperparameters

Start with:

```yaml
retrieval:
  sparse_top_k: 100

reranker:
  batch_size: 32
  threshold: 0.55

answer:
  max_evidence_nodes: 6
  parent_rescue_margin: 0.05
```

Tune in roughly this order:

1. `retrieval.sparse_top_k` (candidate recall versus reranking cost);
2. reranker threshold;
3. max evidence nodes;
4. BM25 field weights/token normalization;
5. reranker model.

---

## 15. Debugging retrieval

Every inference can write a JSONL record containing:

```text
question
sparse_top
reranker_candidate_count
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
-> sparse recall
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

Current tests verify parsing behavior and that sparse Top-K results are passed directly to the reranker.

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

### Why feed BM25 results directly to the reranker?

It keeps the first stage fast and transparent: one Top-K parameter defines the complete reranker workload, with no dense-model memory cost or fusion/expansion changing the candidate count.

### Why deterministic intro by default?

It removes generation variance while retrieval is being tuned. GenAI can be enabled later as a separate ablation.

---

## 19. Suggested next improvements

After the baseline is stable, useful experiments include:

- domain-specific Vietnamese text normalization for BM25;
- query expansion using legal synonyms without altering the answer;
- BM25F-style field weighting experiments for context versus raw text;
- document-aware reranker batching;
- better rule-based structural parsing for specific document families;
- answer-style calibration using train data without using train answers as legal evidence;
- official METEOR scorer integration if released.

Keep these as separate experiments so the effect of hierarchical retrieval remains measurable.
