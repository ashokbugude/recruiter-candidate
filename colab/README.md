# Google Colab — GPU preprocessing

Run offline preprocess on a **CUDA GPU** (fast BGE embeddings). Outputs land in **`artifacts/`** — download and copy into your local repo.

## Quick start

1. Open **`colab/preprocess_gpu.ipynb`** in [Google Colab](https://colab.research.google.com/).
2. **Runtime → Change runtime type → T4 GPU** (or better).
3. Upload this repo to Colab (clone from GitHub, or zip upload).
4. Upload **`challenge/candidates.jsonl`** (see `colab/inputs/README.md`).
5. *(Optional)* Upload existing `artifacts/*` from your PC to skip steps already done.
6. Run all cells.
7. Download **`colab/artifacts_download.zip`** (or zip `artifacts/` manually).
8. Extract into your local **`artifacts/`** folder.

## What Colab runs

| Step | GPU helps? | Output |
|------|------------|--------|
| Setup + silver labels | No | `labels_silver.parquet` |
| Features + BM25 | No | `candidate_features.parquet`, `bm25.pkl` |
| **BGE embeddings** | **Yes (~15–45 min vs 4+ hrs CPU)** | `bge_embeddings.npy`, `faiss.index` |
| **Gemini JD + labels** | No (API; ~$15–25 Flash for 100K) | `jd_requirements.json`, `gemini_tiers.parquet` |

## Gemini API key on Colab

1. Colab sidebar → **Secrets** (key icon) → **Add secret**
   - Name: `GEMINI_API_KEY`
   - Value: your key from [Google AI Studio](https://aistudio.google.com/apikey)
2. Enable **notebook access** for that secret.
3. In the notebook, set **`USE_GEMINI = True`** (default after update).
4. Run all cells — Gemini runs over the network; **only embeddings use CUDA**.

Alternative (no Secrets): the notebook prompts for the key with hidden input (`getpass`).

Optional model overrides (set in notebook before preprocess):

```python
os.environ['REDROB_GEMINI_FLASH_MODEL'] = 'gemini-3.5-flash'
os.environ['REDROB_GEMINI_PRO_MODEL'] = 'gemini-2.5-pro'
```

## Commands (without notebook)

```bash
export GEMINI_API_KEY=your_key   # Colab: use Secrets instead
pip install -r colab/requirements.txt
python colab/run_preprocess_gpu.py --no-skip-llm --force
```

## After Colab (on your PC)

```powershell
# Copy downloaded artifacts into repo artifacts/
python scripts/lock_artifacts.py
python scripts/train_ltr.py
python rank.py --candidates ./challenge/candidates.jsonl --out ./submission.csv
```

Set in `.env` to avoid re-spending Gemini credits:

```
REDROB_SKIP_GEMINI=1
```

## Files in this folder

| File | Purpose |
|------|---------|
| `preprocess_gpu.ipynb` | Main Colab notebook |
| `requirements.txt` | Pinned deps for Colab |
| `run_preprocess_gpu.py` | CLI wrapper used by notebook |
| `inputs/README.md` | What to upload before running |
