"""Domain constants — recycled templates, JD targets, trap patterns."""

from __future__ import annotations

import hashlib
from datetime import date
from typing import Final

# Fixed reference date for reproducible behavioral / honeypot rules (challenge eval window).
REFERENCE_DATE: Final[date] = date(2026, 6, 1)

# Ground-truth-style honeypot rules (~80 subtly impossible profiles in submission spec).
SUBTLE_HONEYPOT_RULES: Final[frozenset[str]] = frozenset({"R1", "R2", "R3", "R5", "R6", "R9"})

SUBTLE_HONEYPOT_WEIGHTS: Final[dict[str, float]] = {
    "R1": 0.45,
    "R2": 0.45,
    "R3": 0.50,
    "R5": 0.45,
    "R6": 0.40,
    "R9": 0.45,
}

# Single subtle rule at or above this weight → tier-0 honeypot.
SUBTLE_HONEYPOT_SINGLE_THRESHOLD: Final[float] = 0.40

JUNIOR_TITLE_KEYWORDS: Final[tuple[str, ...]] = (
    "junior",
    "intern",
    "trainee",
    "entry level",
    "entry-level",
    "graduate",
    "associate ml",
    "associate ai",
)

# ---------------------------------------------------------------------------
# Recycled career-description templates (8 dominant hashes in 100K dataset)
# Each appears ~25K times across career_history entries — honeypot / trap signal.
# ---------------------------------------------------------------------------

KNOWN_TEMPLATE_HASHES: Final[frozenset[str]] = frozenset(
    {
        "1c8d46642c94",  # Enterprise sales / cloud software
        "5468e1b37aa2",  # Customer support team lead
        "03fdb47f3db0",  # Marketing leadership B2B SaaS
        "046bc838535d",  # Business analyst consulting
        "592133a664e0",  # Brand design / creative direction
        "2f3df0fbcafd",  # Mechanical engineering design
        "f97cca022153",  # Senior accounting / month-end close
        "979634cabf2b",  # Content writing / SEO strategy
    }
)

KNOWN_TEMPLATES: Final[tuple[dict[str, str], ...]] = (
    {
        "hash_prefix": "1c8d46642c94",
        "label": "enterprise_sales",
        "snippet": "Enterprise sales of cloud software solutions into the mid-market segment.",
    },
    {
        "hash_prefix": "5468e1b37aa2",
        "label": "customer_support_lead",
        "snippet": "Customer support team lead at a SaaS product.",
    },
    {
        "hash_prefix": "03fdb47f3db0",
        "label": "marketing_leadership",
        "snippet": "Marketing leadership role at a B2B SaaS company.",
    },
    {
        "hash_prefix": "046bc838535d",
        "label": "business_analyst_consulting",
        "snippet": "Business analyst at a consulting firm, working primarily with retail and CPG clients.",
    },
    {
        "hash_prefix": "592133a664e0",
        "label": "brand_design",
        "snippet": "Brand design and creative direction at a consumer-products company.",
    },
    {
        "hash_prefix": "2f3df0fbcafd",
        "label": "mechanical_engineering",
        "snippet": "Mechanical engineering design role at a hardware-product company.",
    },
    {
        "hash_prefix": "f97cca022153",
        "label": "senior_accounting",
        "snippet": "Senior accounting role at a mid-sized company",
    },
    {
        "hash_prefix": "979634cabf2b",
        "label": "content_writing_seo",
        "snippet": "Content writing and SEO strategy for a tech-focused publication.",
    },
)


def career_description_hash(description: str) -> str:
    """MD5 hex digest of normalized career role description."""
    normalized = description.strip()
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def is_known_template(description: str) -> bool:
    """Return True if description matches a known recycled template."""
    prefix = career_description_hash(description)[:12]
    return prefix in KNOWN_TEMPLATE_HASHES


# ---------------------------------------------------------------------------
# JD alignment — Senior AI Engineer @ Redrob
# ---------------------------------------------------------------------------

JD_YOE_MIN: Final[float] = 5.0
JD_YOE_MAX: Final[float] = 9.0
JD_YOE_SOFT_MIN: Final[float] = 4.0
JD_YOE_SOFT_MAX: Final[float] = 12.0

JD_PREFERRED_CITIES: Final[frozenset[str]] = frozenset(
    {"pune", "noida", "delhi", "gurgaon", "gurugram", "mumbai", "hyderabad", "bangalore", "bengaluru"}
)

JD_PRIMARY_HUBS: Final[frozenset[str]] = frozenset({"pune", "noida"})

JD_SECONDARY_HUBS: Final[frozenset[str]] = frozenset(
    {"bangalore", "bengaluru", "gurgaon", "gurugram", "delhi"}
)

# Display order for reasoning (JD emphasizes Pune/Noida first).
JD_HUB_DISPLAY: Final[str] = "Pune/Noida/Bangalore/Delhi-Gurgaon"

CV_SPEECH_SKILL_MARKERS: Final[tuple[str, ...]] = (
    "yolo",
    "asr",
    "speech recognition",
    "speech",
    "computer vision",
    "opencv",
    "image classification",
    "whisper",
    "tts",
    "text-to-speech",
    "audio",
    "object detection",
)

JD_PREFERRED_COUNTRY: Final[str] = "India"

SENIOR_AI_TITLE_KEYWORDS: Final[tuple[str, ...]] = (
    "ai engineer",
    "ml engineer",
    "machine learning engineer",
    "applied scientist",
    "research engineer",
    "search engineer",
    "recommendation engineer",
    "staff ml",
    "staff ai",
    "senior ai",
    "senior ml",
    "nlp engineer",
    "ranking engineer",
)

SEARCH_RETRIEVAL_KEYWORDS: Final[tuple[str, ...]] = (
    "retrieval",
    "ranking",
    "recommendation",
    "embedding",
    "vector",
    "faiss",
    "pinecone",
    "weaviate",
    "elasticsearch",
    "opensearch",
    "bm25",
    "hybrid search",
    "ndcg",
    "learning to rank",
    "ltr",
    "rerank",
    "semantic search",
    "vector search",
    "search engine",
    "search ranking",
    "search infrastructure",
    "information retrieval",
)

ML_PRODUCTION_KEYWORDS: Final[tuple[str, ...]] = (
    "pytorch",
    "tensorflow",
    "transformers",
    "sentence-transformers",
    "bge",
    "fine-tuning",
    "lora",
    "production ml",
    "ml platform",
    "model serving",
    "inference",
)

TRAP_TITLE_KEYWORDS: Final[tuple[str, ...]] = (
    "hr manager",
    "human resources",
    "accountant",
    "content writer",
    "marketing manager",
    "sales executive",
    "business analyst",
    "customer support",
    "operations manager",
    "graphic designer",
    "civil engineer",
    "mechanical engineer",
    "project manager",
)

CONSULTING_FIRMS: Final[frozenset[str]] = frozenset(
    {
        "tcs",
        "tata consultancy",
        "infosys",
        "wipro",
        "accenture",
        "cognizant",
        "capgemini",
        "hcl",
        "tech mahindra",
        "mindtree",
        "ltimindtree",
        "mphasis",
        "persistent",
        "cyient",
        "deloitte consulting",
        "ey",
        "kpmg",
        "pwc",
    }
)

PRODUCT_COMPANY_INDICATORS: Final[tuple[str, ...]] = (
    "saas",
    "product",
    "platform",
    "startup",
    "series",
    "fintech",
    "e-commerce",
    "ecommerce",
    "marketplace",
    "consumer",
    "b2b software",
    "edtech",
    "healthtech",
)

AI_SKILL_NAMES: Final[frozenset[str]] = frozenset(
    {
        "nlp",
        "pytorch",
        "tensorflow",
        "transformers",
        "rag",
        "llm",
        "fine-tuning llms",
        "embeddings",
        "vector search",
        "faiss",
        "pinecone",
        "weaviate",
        "elasticsearch",
        "opensearch",
        "learning to rank",
        "ranking",
        "recommendation systems",
        "semantic search",
        "sentence-transformers",
        "bge",
        "xgboost",
        "lightgbm",
        "mlflow",
        "kubeflow",
        "mlops",
        "model deployment",
        "image classification",
        "computer vision",
    }
)


def is_junior_title(title: str, headline: str = "") -> bool:
    """True when title/headline indicates non-senior level."""
    combined = f"{title} {headline}".lower()
    return any(keyword in combined for keyword in JUNIOR_TITLE_KEYWORDS)


def matches_senior_ai_title(title: str, headline: str = "") -> bool:
    """Senior AI title match excluding junior/intern patterns."""
    if is_junior_title(title, headline):
        return False
    combined = f"{title} {headline}".lower()
    return any(keyword in combined for keyword in SENIOR_AI_TITLE_KEYWORDS)


def count_search_retrieval_hits(text: str) -> int:
    """Count IR keyword hits in profile text."""
    lowered = text.lower()
    return sum(1 for keyword in SEARCH_RETRIEVAL_KEYWORDS if keyword in lowered)
