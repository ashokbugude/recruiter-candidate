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
python scripts/lock_artifacts.py     # prevent accidental Gemini re-runs
```

## Rank & submit

```bash
python rank.py --candidates ./challenge/candidates.jsonl --out ./submission.csv
```

Requires `artifacts/`: features, BM25, FAISS embeddings, `ltr_model.lgb` (built by preprocess + train).

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
