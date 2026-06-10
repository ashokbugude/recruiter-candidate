FROM python:3.12-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY rank.py ./
COPY scripts ./scripts
COPY challenge/validate_submission.py ./challenge/validate_submission.py
COPY challenge/sample_candidates.json ./challenge/sample_candidates.json
COPY challenge/candidate_schema.json ./challenge/candidate_schema.json
COPY artifacts ./artifacts

# Fail fast if rank-ready tuning artifacts are missing from the image.
RUN test -f artifacts/fusion_params.json \
    && test -f artifacts/modifier_params.json \
    && test -f artifacts/career_recall_scores.json \
    && test -f artifacts/ltr_model.lgb

ENV PYTHONUNBUFFERED=1
ENV REDROB_ARTIFACTS_DIR=/app/artifacts
ENV REDROB_CANDIDATES_PATH=/app/challenge/candidates.jsonl

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
