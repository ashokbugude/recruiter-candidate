# Sarva Automata — Methodology Document
**Redrob IndiaRuns Track 1 · Intelligent Candidate Ranking**

| Field | Value |
|-------|-------|
| Team | Sarva Automata |
| Team lead | Arjun Mangarath |
| GitHub | https://github.com/ashokbugude/recruiter-candidate |
| Sandbox | https://huggingface.co/spaces/ashokbugude/redrob-ranker |
| Portal CSV | `team_sarva_automata.csv` |

---

## 1. Problem & approach

**Challenge:** Rank 100,000 candidates for a Senior AI Engineer role (search / retrieval / ranking) with CPU-only inference, no network at rank time, and adversarial profiles (honeypots, keyword stuffers, behavioral twins).

**Solution:** A three-stage retrieval–rank funnel that mirrors production recruiting systems:

1. **Hybrid recall** — BM25 + BGE dense embeddings + career RRF → ~2,000 candidate pool  
2. **LightGBM LambdaRank** — 45+ hand-crafted features, trap penalties, behavioral multipliers  
3. **BGE cross-encoder rerank** — top-700 pairs rescored; CE / LTR / RRF fusion with Optuna-tuned weights  

Post-fusion modifiers enforce availability in the top 30, cap template-clone summaries, and down-rank research titles without production evidence.

**Differentiation:** IR-style retrieval + learned ranking + Redrob behavioral signals as first-class signals—not keyword matching. Explicit honeypot and trap handling per challenge documentation.

---

## 2. JD understanding

Key requirements extracted from the job description:

- 5–8 years production ML / applied AI (not tutorial-only LLM wrappers)  
- Search, retrieval, and ranking systems at scale  
- Embeddings, vector search, hybrid retrieval  
- India hubs (Pune, Noida, Bangalore, Delhi-Gurgaon)  
- Product engineering over pure research  
- Recruiter reachability (response rate, notice period, engagement)

**Signals beyond keywords:** silver-tier proxy labels, career-history evidence, GitHub activity, trap flags, and 23 Redrob behavioral signals (response rate, recency, notice, etc.).

---

## 3. Ranking methodology

### Retrieval & scoring

| Stage | Method | Output |
|-------|--------|--------|
| Recall | BM25 (3K) + FAISS dense BGE (3K) + career RRF | ~2K pool |
| LTR | LightGBM LambdaRank + trap / behavior penalties | Scored pool |
| Rerank | BGE-reranker-base on top-700 | Semantic fit |
| Fusion | Weighted CE + LTR + RRF (Optuna-tuned) | Fused scores |
| Modifiers | Availability cap, clone limit, research penalty | Final order |

### Models & heuristics

- BM25, FAISS, LightGBM LambdaRank, BGE-reranker-base  
- 12+ honeypot / trap heuristics; hard-exclude impossible profiles  
- Optuna tuning on silver-proxy NDCG@10 (modifiers + fusion)  
- Gemini used **offline only** for JD parsing and archetype labeling (not at rank time)

### Signal combination

```
final_order = modifiers(
  fuse(CE, LTR, RRF) × behavioral_multiplier − trap_penalties
)
```

Submission scores are rank-monotonic, derived from fused model scores (capped at 0.9999).

---

## 4. Explainability

- **Reasoning column:** 5–6 rank-band templates citing YOE, skills, location, availability, production vs research, and evaluation metrics from **whitelisted fields only**  
- **No runtime LLM** — prevents hallucinated justifications  
- **Explicit gap flags** — notice period, low response rate, outside-India location, CV/speech stack mismatch  
- **Audit script** — `scripts/audit_submission.py` reports tier-5 concentration, honeypots, clones, availability in top ranks

---

## 5. Data quality & traps

| Risk | Mitigation |
|------|------------|
| Honeypots (~80 subtle impossible profiles) | Hard-exclude + subtractive trap penalties |
| Keyword stuffers | LTR + CE semantic scoring; skill trust multipliers |
| Behavioral twins (same summary) | Template-clone detection; cap in top 30 |
| Research-only titles | Penalty without production keywords |
| Low availability | Steep down-rank; top-30 availability cap |

---

## 6. End-to-end workflow

**Offline (~90 min, may use GPU / Gemini):**

```
preprocess.py → features, BM25, FAISS, silver labels
train_ltr.py → ltr_model.lgb
tune_modifiers.py → fusion_params.json, modifier_params.json
```

**Online (CPU, no network, ≤5 min for 100K):**

```
rank.py --candidates candidates.jsonl --out team_sarva_automata.csv
  → hybrid recall → LTR → CE rerank → fusion + modifiers → CSV + audit
```

**Sandbox (§10.5):** Hugging Face Docker Space — `candidates.jsonl` pre-bundled in Docker; UI shows it on load; one-click live `rank.py` (~5–15 min CPU) → `team_sarva_automata.csv`.

---

## 7. Results (submission audit)

Audit on `team_sarva_automata.csv`:

| Metric | Value |
|--------|-------|
| Proxy NDCG@10 | **1.0** |
| NDCG@30 | 0.9922 |
| Tier-5 in top 10 / 30 / 100 | **9 / 26 / 70** |
| Honeypots in top 100 | **0** |
| Template clones in top 30 | 8 (within cap) |
| Low availability in top 30 | **0** |

**Compute:** 16 GB RAM, CPU-only at inference, no API calls during ranking. Embeddings and indexes precomputed offline.

---

## 8. Technology stack

Python 3.12 · Polars · LightGBM · sentence-transformers · FAISS · rank-bm25 · FastAPI · Optuna · Gemini (offline JD/labels) · Cursor (engineering assist)

---

## 9. Submission assets

| Asset | Location |
|-------|----------|
| Code | https://github.com/ashokbugude/recruiter-candidate |
| Ranked output | `team_sarva_automata.csv` (portal upload) |
| Reproduce | `python rank.py --candidates ./challenge/candidates.jsonl --out ./team_sarva_automata.csv` |
| Sandbox | https://huggingface.co/spaces/ashokbugude/redrob-ranker |
| Metadata | `submission_metadata.yaml` |
| This document | `docs/METHODOLOGY.md` · deck: `challenge/Idea Submission Template _ Redrob.pptx` |

**AI tools (declared):** Cursor for architecture and code; Gemini offline for JD parse and labeling only. No candidate data sent to LLMs during ranking.

---

*Export to PDF: open `challenge/Idea Submission Template _ Redrob.pptx` in PowerPoint → Save as PDF, or print this file from VS Code / browser.*
