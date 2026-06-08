# Google Colab — GPU preprocessing

Run offline preprocess on a **CUDA GPU** (fast BGE embeddings). Outputs land in **`artifacts/`** — download and copy into your local repo.

## Quick start

1. Open **`colab/preprocess_gpu.ipynb`** in [Google Colab](https://colab.research.google.com/).
2. **Runtime → Change runtime type → T4 GPU**
3. Run all cells (notebook clones from GitHub)
4. Upload **`challenge/candidates.jsonl`** when prompted (~465 MB, not in git)
5. Provide Gemini credentials (see below)
6. Download **`colab/artifacts_download.zip`** → extract into local **`artifacts/`**

## Gemini authentication (ADC)

The notebook uses **Application Default Credentials** via **Vertex AI** (not the Developer API key flow). Your `authorized_user` JSON with `refresh_token` must include `quota_project_id` (GCP project with Vertex AI enabled).

**Option A — Colab Secret (recommended):**

1. Sidebar → **Secrets** → add **`GOOGLE_ADC_JSON`**
2. Paste the full JSON (fields: `client_id`, `client_secret`, `quota_project_id`, `refresh_token`, `type`, `universe_domain`)
3. Enable notebook access

**Option B — file upload:** run the auth cell and upload your credentials JSON when prompted.

Locally, save the same JSON as `credentials/adc.json` and set:

```
GOOGLE_APPLICATION_CREDENTIALS=./credentials/adc.json
```

**Fallback:** `GEMINI_API_KEY` still works if you prefer an API key.

## What Colab runs

| Step | GPU helps? | Output |
|------|------------|--------|
| Setup + silver labels | No | `labels_silver.parquet` |
| Features + BM25 | No | `candidate_features.parquet`, `bm25.pkl` |
| **BGE embeddings** | **Yes (~15–45 min)** | `bge_embeddings.npy`, `faiss.index` |
| **Gemini JD + labels** | No (API) | `jd_requirements.json`, `gemini_tiers.parquet` |

## After Colab (on your PC)

```powershell
python scripts/lock_artifacts.py
python scripts/train_ltr.py
python rank.py --candidates ./challenge/candidates.jsonl --out ./submission.csv
```

Set `REDROB_SKIP_GEMINI=1` in `.env` to avoid re-calling Gemini locally.

## Files

| File | Purpose |
|------|---------|
| `preprocess_gpu.ipynb` | Main Colab notebook |
| `requirements.txt` | Colab dependencies |
| `run_preprocess_gpu.py` | CLI wrapper |
| `inputs/README.md` | Upload checklist |
