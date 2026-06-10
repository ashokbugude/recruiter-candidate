"""Central artifact filename constants."""

from __future__ import annotations

FUSION_PARAMS = "fusion_params.json"
MODIFIER_PARAMS = "modifier_params.json"
CAREER_SCORES = "career_recall_scores.json"
LABELS_SILVER = "labels_silver.parquet"
LTR_MODEL = "ltr_model.lgb"
TUNING_REPORT_MODIFIERS = "tuning_report_modifiers.json"
SUBMISSION_AUDIT = "submission_audit.json"

RANK_REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "job_description.txt",
    "jd_requirements.json",
    "candidate_features.parquet",
    "bm25.pkl",
    "faiss.index",
    "bge_embeddings.npy",
    "candidate_id_index.json",
    LTR_MODEL,
    FUSION_PARAMS,
    MODIFIER_PARAMS,
    CAREER_SCORES,
    LABELS_SILVER,
)

RANK_OPTIONAL_ARTIFACTS: tuple[str, ...] = (
    "gemini_tiers.parquet",
    "best_params.json",
    TUNING_REPORT_MODIFIERS,
)
