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

## Try the API

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Artifact check |
| `GET /rank/sample` | Rank bundled sample candidates (≤100) |
| `POST /rank/upload` | Upload JSONL (max 500 lines) |
| `GET /docs` | Swagger UI |

## Full submission (local / Stage 3)

```bash
python rank.py --candidates ./challenge/candidates.jsonl --out ./team_sarva_automata.csv
```

GitHub: [recruiter-candidate](https://github.com/ashokbugude/recruiter-candidate)
