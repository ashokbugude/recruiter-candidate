# Google Colab — GPU preprocessing

Run offline preprocess on a **CUDA GPU** (fast BGE embeddings). Outputs land in **`artifacts/`** — download and copy into your local repo.

## Quick start

1. Open **`colab/preprocess_gpu.ipynb`** in [Google Colab](https://colab.research.google.com/).
2. **Runtime → Change runtime type → T4 GPU**
3. Run all cells (notebook clones from GitHub)
4. Upload **`challenge/candidates.jsonl`** when prompted (~465 MB, not in git)
5. Set Colab Secret **`GEMINI_API_KEY`**
6. Download **`colab/artifacts_download.zip`** → extract into local **`artifacts/`**

## What Colab runs

| Step | GPU helps? | Output |
|------|------------|--------|
| Setup + silver labels | No | `labels_silver.parquet` |
| Features + BM25 | No | `candidate_features.parquet`, `bm25.pkl` |
| **BGE embeddings** | **Yes (~15–45 min)** | `bge_embeddings.npy`, `faiss.index` |
| **Gemini JD + labels** | No (API) | `jd_requirements.json`, `gemini_tiers.parquet` |

## Gemini API key

1. Colab sidebar → **Secrets** → add **`GEMINI_API_KEY`**
2. Enable notebook access
3. In notebook: `USE_GEMINI = True`

## After Colab (on your PC)

```powershell
python scripts/lock_artifacts.py
python scripts/train_ltr.py
python rank.py --candidates ./challenge/candidates.jsonl --out ./submission.csv
```

Set `REDROB_SKIP_GEMINI=1` in `.env` to avoid re-spending Gemini credits locally.

## Files

| File | Purpose |
|------|---------|
| `preprocess_gpu.ipynb` | Main Colab notebook |
| `requirements.txt` | Colab dependencies |
| `run_preprocess_gpu.py` | CLI wrapper |
| `inputs/README.md` | Upload checklist |
