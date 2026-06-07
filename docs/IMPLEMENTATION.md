# Implementation Plan — Step by Step

Production-grade delivery for the Redrob IndiaRuns candidate ranking system.

---

## Phase 0 — Project Setup & Foundation

| Sub-step | Task | Output | Status |
|----------|------|--------|--------|
| **0.1** | Repository layout & Python package structure | `app/`, `scripts/`, `artifacts/`, `tests/`, `prompts/` | ✅ |
| **0.2** | Dependency management (pinned versions) | `requirements.txt`, `pyproject.toml` | ✅ |
| **0.3** | Configuration & secrets handling | `app/config.py`, `.env.example`, `.gitignore` | ✅ |
| **0.4** | Artifact bootstrap (JD, specs) | `artifacts/*.txt`, `scripts/setup_artifacts.py` | ✅ |
| **0.5** | Entry point stub & logging | `rank.py`, `app/logging_setup.py` | ✅ |
| **0.6** | Documentation & submission metadata | `README.md`, `submission_metadata.yaml` | ✅ |
| **0.7** | Phase 0 verification tests | `tests/test_phase0_setup.py` | ✅ |

**Exit criteria:** `python -m pytest tests/test_phase0_setup.py` passes; `python rank.py --help` works.

---

## Phase 1 — EDA & Silver Labels

| Sub-step | Task | Output | Status |
|----------|------|--------|--------|
| 1.1 | Dataset statistics script | `scripts/eda.py`, `artifacts/eda_report.json` | ✅ |
| 1.2 | Known template hash registry | `app/constants.py` → `KNOWN_TEMPLATES` | ✅ |
| 1.3 | Senior AI / honeypot candidate lists | `artifacts/candidate_lists.json` | ✅ |
| 1.4 | Manual + heuristic silver labels | `artifacts/labels_silver.parquet` | ✅ |
| 1.5 | Label quality validation | `tests/test_silver_labels.py` | ✅ |

**Exit criteria:** `python scripts/eda.py` and `python scripts/build_silver_labels.py` complete; `pytest tests/test_silver_labels.py` passes.

---

## Phase 2 — Offline Preprocessing

| Sub-step | Task | Output | Status |
|----------|------|--------|--------|
| **2.1** | Gemini JD parser + prompt | `prompts/parse_jd.txt`, `scripts/parse_jd.py`, `app/jd_requirements.py` | ✅ |
| **2.2** | 45-feature extractor | `app/features.py`, `app/feature_store.py` | ✅ |
| **2.3** | BGE embeddings + FAISS index | `app/embeddings.py`, `artifacts/bge_embeddings.npy`, `faiss.index` | ✅ |
| **2.4** | BM25 index | `app/bm25_index.py`, `artifacts/bm25.pkl` | ✅ |
| **2.5** | Gemini archetype batch labeler | `prompts/label_archetype.txt`, `scripts/label_archetypes.py` | ✅ |
| **2.6** | Preprocess orchestrator | `scripts/preprocess.py`, `tests/test_phase2_preprocess.py` | ✅ |

**Exit criteria:** `python scripts/preprocess.py --limit 100` succeeds; `pytest tests/test_phase2_preprocess.py` passes.

---

## Phase 3 — Train Learning-to-Rank

| Sub-step | Task | Output |
|----------|------|--------|
| 3.1 | LightGBM LambdaRank training | `scripts/train_ltr.py`, `artifacts/ltr_model.lgb` |
| 3.2 | Optuna hyperparameter tuning | `scripts/tune.py`, `artifacts/best_params.json` |
| 3.3 | Proxy NDCG@10 validation | `artifacts/tuning_report.json` |

---

## Phase 4 — 3-Stage Ranking Pipeline

| Sub-step | Task | Output |
|----------|------|--------|
| 4.1 | Hybrid recall (BM25 + FAISS + RRF) | `app/recall.py` |
| 4.2 | Trap & honeypot detection | `app/traps.py` |
| 4.3 | Behavioral signal modifier | `app/behavioral.py` |
| 4.4 | LTR scorer integration | `app/ltr.py` |
| 4.5 | Cross-encoder reranker | `app/reranker.py` |
| 4.6 | Pipeline orchestrator | `app/pipeline.py` |
| 4.7 | CLI & submission CSV writer | `rank.py`, `app/reasoning.py` |

---

## Phase 5 — Reasoning & QA

| Sub-step | Task | Output |
|----------|------|--------|
| 5.1 | Fact-anchored reasoning generator | `app/reasoning.py` |
| 5.2 | Honeypot audit on top 100 | `scripts/audit_submission.py` |
| 5.3 | Reproduction test on 16GB CPU | documented in README |

---

## Phase 6 — API, Docker & Submission

| Sub-step | Task | Output |
|----------|------|--------|
| 6.1 | FastAPI service | `app/main.py` |
| 6.2 | Dockerfile | `Dockerfile` |
| 6.3 | Sandbox deployment | HuggingFace Spaces URL |
| 6.4 | PDF deck & final submission | portal upload |
