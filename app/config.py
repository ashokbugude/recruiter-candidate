"""Central configuration — paths, constraints, and environment settings."""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.constants import REFERENCE_DATE

# Repository root (parent of app/)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment and defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="REDROB_",
        extra="ignore",
    )

    # Paths
    artifacts_dir: Path = Field(default=PROJECT_ROOT / "artifacts")
    challenge_dir: Path = Field(default=PROJECT_ROOT / "challenge")
    candidates_path: Path = Field(default=PROJECT_ROOT / "challenge" / "candidates.jsonl")
    sample_candidates_path: Path = Field(
        default=PROJECT_ROOT / "challenge" / "sample_candidates.json"
    )
    submission_csv_path: Path = Field(default=PROJECT_ROOT / "team_sarva_automata.csv")
    job_description_path: Path = Field(default=PROJECT_ROOT / "artifacts" / "job_description.txt")
    jd_requirements_path: Path = Field(default=PROJECT_ROOT / "artifacts" / "jd_requirements.json")

    # Ranking constraints (submission spec §3)
    max_runtime_seconds: int = Field(default=300, ge=1)
    max_ram_gb: int = Field(default=16, ge=1)
    top_k_output: int = Field(default=100, ge=1, le=100)
    recall_pool_size: int = Field(default=2000, ge=100)
    rerank_pool_size: int = Field(default=400, ge=10)
    top_availability_cap: int = Field(
        default=30,
        ge=1,
        le=100,
        description="Ranks 1..N must prefer high-availability candidates when possible.",
    )

    # Logging
    log_level: str = Field(default="INFO")

    # Offline LLM (preprocessing only)
    gemini_api_key: str | None = Field(default=None)
    gemini_flash_model: str = Field(default="gemini-2.5-flash")
    gemini_label_batch_size: int = Field(default=50, ge=1, le=200)
    gemini_label_sleep_seconds: float = Field(default=0.2, ge=0.0, le=10.0)
    gemini_pro_model: str = Field(default="gemini-2.5-pro")
    # Used when authenticating via ADC (OAuth / service account) → Vertex AI
    google_cloud_project: str | None = Field(default=None)
    google_cloud_location: str = Field(default="global")
    gemini_jd_parse: bool = Field(
        default=True,
        description="Call Gemini Pro for JD parse when credentials are available.",
    )
    gemini_labels: bool = Field(
        default=False,
        description="Call Gemini Flash for 100K archetype labels (slow; off by default).",
    )
    skip_gemini: bool = Field(
        default=False,
        description="Master switch: if true, disables both gemini_jd_parse and gemini_labels.",
    )

    # Ranking blend weights (Stage 3 fusion)
    rerank_ce_weight: float = Field(default=0.55, ge=0.0, le=1.0)
    rerank_ltr_weight: float = Field(default=0.30, ge=0.0, le=1.0)
    rerank_rrf_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1)
    career_rrf_weight: float = Field(default=0.65, ge=0.0, le=1.0)
    bm25_recall_k: int = Field(default=3000, ge=100)
    dense_recall_k: int = Field(default=3000, ge=100)

    # Reproducible date for behavioral / honeypot rules
    reference_date: date = Field(default=REFERENCE_DATE)

    @field_validator(
        "artifacts_dir",
        "challenge_dir",
        "candidates_path",
        "sample_candidates_path",
        "submission_csv_path",
        "job_description_path",
        "jd_requirements_path",
        mode="before",
    )
    @classmethod
    def resolve_path(cls, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def ensure_artifacts_dir(self) -> Path:
        """Create artifacts directory if missing."""
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        return self.artifacts_dir

    def artifact_path(self, name: str) -> Path:
        """Resolve a file path inside the artifacts directory."""
        return self.artifacts_dir / name


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
