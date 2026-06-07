# Colab inputs checklist

Upload these into the Colab runtime **before** running the notebook (via Drive mount or file upload).

## Required

| File | Colab path | Size (approx) |
|------|------------|---------------|
| `candidates.jsonl` | `challenge/candidates.jsonl` | ~500 MB |

## Optional (copy from your PC to skip slow steps)

If you already ran preprocess locally, upload these into `artifacts/` to **only run GPU embeddings**:

| File | Notes |
|------|--------|
| `labels_silver.parquet` | From `scripts/build_silver_labels.py` |
| `gemini_tiers.parquet` | Optional Gemini/silver tiers |
| `candidate_features.parquet` | 100K feature rows |
| `jd_requirements.json` | Parsed JD |
| `bm25.pkl` | BM25 index |
| `job_description.txt` | From `setup_artifacts.py` |

## Optional Gemini on Colab

Set `GEMINI_API_KEY` in Colab Secrets (see `colab/README.md`). Labels cost ~$15–25 (Flash); embeddings still use CUDA locally in Colab.

## Not needed in Colab

- Local `.env` file — use Colab **Secrets** (`GEMINI_API_KEY`) instead
- `venv/` — Colab has its own Python env.

## After Colab finishes

Download **`artifacts/`** (especially `bge_embeddings.npy`, `faiss.index`, `candidate_id_index.json`) and copy into this repo’s `artifacts/` folder on your PC.

Then locally:

```powershell
python scripts/lock_artifacts.py
python scripts/train_ltr.py
python rank.py --candidates ./challenge/candidates.jsonl --out ./submission.csv
```
