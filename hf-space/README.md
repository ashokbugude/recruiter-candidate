---
title: Redrob Candidate Ranker
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Redrob IndiaRuns — Sarva Automata

Hybrid recall (BM25 + BGE dense + career RRF) → LightGBM LTR → BGE cross-encoder rerank.
CPU-only, offline artifacts, no network at inference.

## Hardware

Use **CPU** (not GPU) on Hugging Face Spaces. **16 GB RAM** recommended — sample ranking
loads full FAISS/embeddings (~100K) but only ranks ≤100 candidates.

First Docker build is **2–4+ GB** and may take **20–40 minutes** (artifacts + pip + models).

## Sandbox (submission_spec §10.5)

Open the Space home page (`/`). **`candidates.jsonl` (100K rows) is pre-loaded in the Docker image** — shown as ready on page load, no file picker needed.

Click **Rank and download CSV** to run live `rank.py` on the bundled pool (~5–15 min CPU).

**Optional:** upload your own JSONL / JSON array to rank that file instead (overrides bundled pool). Full `candidates.jsonl` upload supported up to 100K rows.

| Endpoint | Description |
|----------|-------------|
| `GET /` | Pre-loaded pool + rank button (§10.5 demo UI) |
| `GET /pool` | Bundled `candidates.jsonl` metadata |
| `GET /health` | Artifact + pool check |
| `POST /rank/run` | Run `rank.py` on bundled pool → CSV download |
| `POST /rank/upload` | Rank uploaded file (overrides bundled pool) |
| `GET /rank/sample` | API alias → `POST /rank/run` |
| `GET /docs` | Swagger UI |

CSV columns: `candidate_id,rank,score,reasoning` (submission_spec §2).

## Full submission (local / Stage 3)

```bash
python rank.py --candidates ./challenge/candidates.jsonl --out ./team_sarva_automata.csv
```

GitHub: [recruiter-candidate](https://github.com/ashokbugude/recruiter-candidate)
