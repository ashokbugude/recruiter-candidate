"""JD requirements schema, heuristic parser, and Gemini-backed extraction."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from app.constants import (
    JD_PREFERRED_CITIES,
    JD_PREFERRED_COUNTRY,
    JD_YOE_MAX,
    JD_YOE_MIN,
    ML_PRODUCTION_KEYWORDS,
    SEARCH_RETRIEVAL_KEYWORDS,
    SENIOR_AI_TITLE_KEYWORDS,
    TRAP_TITLE_KEYWORDS,
)
from app.gemini_client import (
    PRO_MODEL_FALLBACKS,
    generate_json,
    has_gemini_auth,
    load_prompt_template,
    resolve_pro_model,
)

logger = logging.getLogger(__name__)


class JDRequirements(BaseModel):
    """Structured requirements extracted from the job description."""

    role_title: str = "Senior AI Engineer"
    must_have_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    must_have_keywords: list[str] = Field(default_factory=list)
    target_titles: list[str] = Field(default_factory=list)
    disqualifiers: list[str] = Field(default_factory=list)
    yoe_min: float = JD_YOE_MIN
    yoe_max: float = JD_YOE_MAX
    preferred_country: str = JD_PREFERRED_COUNTRY
    preferred_cities: list[str] = Field(default_factory=lambda: sorted(JD_PREFERRED_CITIES))
    source: str = "heuristic"


def build_heuristic_requirements(jd_text: str) -> JDRequirements:
    """Build JD requirements from constants + JD text (no API call)."""
    must_skills = sorted(
        {
            "python",
            "embeddings",
            "retrieval",
            "ranking",
            "vector search",
            "faiss",
            "sentence-transformers",
            "bge",
            "hybrid search",
            "bm25",
            "ndcg",
            "learning to rank",
            "pytorch",
            "evaluation framework",
        }
    )
    nice_skills = sorted(
        {
            "llm fine-tuning",
            "lora",
            "xgboost",
            "lightgbm",
            "pinecone",
            "weaviate",
            "opensearch",
            "mlops",
            "a/b testing",
        }
    )
    keywords = sorted(set(SEARCH_RETRIEVAL_KEYWORDS) | set(ML_PRODUCTION_KEYWORDS))
    target_titles = sorted(set(SENIOR_AI_TITLE_KEYWORDS))
    disqualifiers = sorted(
        {
            "keyword stuffing",
            "consulting-only career",
            "pure research without production",
            "langchain-only recent ai",
            "title-chaser job hopping",
            "inactive profile",
        }
    )
    if "consulting" in jd_text.lower():
        disqualifiers.append("consulting firms only")

    return JDRequirements(
        must_have_skills=must_skills,
        nice_to_have_skills=nice_skills,
        must_have_keywords=keywords,
        target_titles=target_titles,
        disqualifiers=disqualifiers,
        source="heuristic",
    )


def parse_jd_with_gemini(jd_text: str, *, settings) -> JDRequirements:
    """Parse JD using Gemini Pro; falls back to heuristic on failure."""
    template = load_prompt_template("parse_jd.txt")
    prompt = template.replace("{{JOB_DESCRIPTION}}", jd_text)
    try:
        payload = generate_json(
            prompt,
            settings=settings,
            model=resolve_pro_model(settings),
            temperature=0.0,
            model_fallbacks=PRO_MODEL_FALLBACKS,
        )
        if isinstance(payload, dict):
            payload["source"] = "gemini_pro"
            return JDRequirements.model_validate(payload)
    except Exception as exc:
        logger.warning("Gemini JD parse failed (%s); using heuristic fallback", exc)
    return build_heuristic_requirements(jd_text)


def load_or_build_jd_requirements(
    jd_path: Path,
    output_path: Path,
    *,
    settings,
    force: bool = False,
) -> JDRequirements:
    """Load cached requirements or build from JD file."""
    if output_path.exists() and not force:
        logger.info("Loading cached JD requirements from %s", output_path)
        return JDRequirements.model_validate(json.loads(output_path.read_text(encoding="utf-8")))

    jd_text = jd_path.read_text(encoding="utf-8")
    if has_gemini_auth(settings):
        logger.info("Parsing JD with Gemini Pro")
        requirements = parse_jd_with_gemini(jd_text, settings=settings)
    else:
        logger.warning("No Gemini credentials — using heuristic JD requirements")
        requirements = build_heuristic_requirements(jd_text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(requirements.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Wrote JD requirements to %s (source=%s)", output_path, requirements.source)
    return requirements


def normalize_skill(name: str) -> str:
    return name.strip().lower()


def skill_matches_jd(skill_name: str, jd: JDRequirements) -> bool:
    """Fuzzy match skill name against JD must/nice lists."""
    from rapidfuzz import fuzz

    normalized = normalize_skill(skill_name)
    pool = [normalize_skill(s) for s in jd.must_have_skills + jd.nice_to_have_skills]
    if normalized in pool:
        return True
    return any(fuzz.partial_ratio(normalized, target) >= 85 for target in pool)


def title_matches_jd_targets(title: str, jd: JDRequirements) -> bool:
    lowered = title.lower()
    return any(target in lowered for target in jd.target_titles)


def is_trap_title(title: str) -> bool:
    lowered = title.lower()
    return any(trap in lowered for trap in TRAP_TITLE_KEYWORDS)
