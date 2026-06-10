# Redrob IndiaRuns — Intelligent Candidate Ranking

Rank 100,000 candidate profiles against the Redrob Senior AI Engineer job description using hybrid retrieval, LightGBM LTR, and cross-encoder reranking.

## Requirements

- Python 3.11+
- 16 GB RAM at rank time (CPU only, no network)
- `challenge/candidates.jsonl` from the hackathon bundle ([see challenge/DATA.md](challenge/DATA.md))

## Quick start (local)

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

python scripts/setup_artifacts.py
python scripts/build_silver_labels.py
copy .env.example .env         # set GOOGLE_APPLICATION_CREDENTIALS or GEMINI_API_KEY (offline only)

python -m pytest tests/ -q
```

## Preprocess (once)

**Colab GPU (recommended):** see [colab/README.md](colab/README.md)

```bash
# Local or Colab after clone
python scripts/preprocess.py --skip-llm          # no Gemini spend
python scripts/preprocess.py --force-llm --force # refresh Gemini labels

python scripts/tune.py --trials 50   # optional
python scripts/train_ltr.py
python scripts/build_career_recall_scores.py   # or: preprocess.py --step career_recall
python scripts/tune_modifiers.py --trials 120  # fusion + modifier tuning + live verify
python scripts/lock_artifacts.py     # prevent accidental Gemini re-runs
```

## Rank & submit

```bash
python rank.py --candidates ./challenge/candidates.jsonl --out ./team_sarva_automata.csv
python scripts/audit_submission.py team_sarva_automata.csv
python challenge/validate_submission.py team_sarva_automata.csv
```

Requires `artifacts/`: features, BM25, FAISS embeddings, `ltr_model.lgb` (built by preprocess + train).

## Pre-upload verification

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/verify_submission.py
python scripts/verify_submission.py --fingerprint   # key rank positions
python scripts/compare_submissions.py old.csv new.csv
```

Cross-encoder scoring uses fixed seeds (`torch.manual_seed(42)`, `np.random.seed(42)`) for tune vs live parity. Use `scripts/tune_modifiers.py` (not deprecated `tune_fusion.py`).

## Sandbox (Docker / FastAPI / Hugging Face)

```bash
docker build -t redrob-ranker .
docker run -p 7860:7860 redrob-ranker
# GET http://localhost:7860/health
# GET http://localhost:7860/rank/sample
```

### Hugging Face Spaces (portal sandbox link)

HF Docker Spaces need the **full `artifacts/` tree** (not in main GitHub — gitignored). Package and push:

```bash
python scripts/package_hf_space.py --out ./dist/hf-space
cd dist/hf-space
git lfs install
git init
git remote add origin https://huggingface.co/spaces/<YOUR_USER>/redrob-ranker
git add .
git commit -m "HF Docker Space with offline artifacts"
git push
```

- Use **CPU** hardware (16 GB RAM). Image is ~2–4 GB; first build ~20–40 min.
- Space README template: `hf-space/README.md` (SDK docker, port 7860).
- Cross-encoder weights are baked in `Dockerfile` at build time (no network at rank).

Local without Docker: `uvicorn app.main:app --host 0.0.0.0 --port 7860`

## Clone in Google Colab

```python
!git clone https://github.com/ashokbugude/recruiter-candidate.git
%cd recruiter-candidate
# Runtime → T4 GPU → open colab/preprocess_gpu.ipynb
```

Upload `candidates.jsonl` to `challenge/` (not in git). Set Colab Secret `GOOGLE_ADC_JSON` (ADC) or `GEMINI_API_KEY` if using Gemini.

## Project layout

```
app/           ranking pipeline (recall, LTR, reranker, traps)
scripts/       preprocess, train, tune
artifacts/     generated indexes & models (gitignored binaries)
challenge/     hackathon data & schema
colab/         GPU + Gemini notebook
prompts/       Gemini prompt templates (offline only)
rank.py        submission CLI
```

## Security

- Never commit `.env` or API keys
- Gemini is **offline preprocess only** — `rank.py` makes no API calls
- Generated artifacts are gitignored; download from Colab and keep locally

## Docs

- [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) — build phases
- [docs/TECHNICAL_PLAN.md](docs/TECHNICAL_PLAN.md) — architecture

## License

MIT
