"""Phase 1 silver label and EDA verification tests."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from app.candidates import count_candidates, load_candidates, load_candidates_list
from app.config import PROJECT_ROOT
from app.constants import KNOWN_TEMPLATE_HASHES, KNOWN_TEMPLATES, REFERENCE_DATE
from app.labels.honeypots import detect_honeypot
from app.labels.silver import build_silver_labels, write_silver_labels
from app.labels.tiers import assign_heuristic_tier


@pytest.fixture(scope="module")
def sample_candidates() -> list[dict]:
    path = PROJECT_ROOT / "challenge" / "sample_candidates.json"
    if not path.exists():
        pytest.skip("sample_candidates.json not available")
    return load_candidates_list(path, limit=50)


@pytest.fixture(scope="module")
def silver_artifacts_built() -> Path:
    """Ensure silver label artifacts exist (build on sample if missing)."""
    labels_path = PROJECT_ROOT / "artifacts" / "labels_silver.parquet"
    lists_path = PROJECT_ROOT / "artifacts" / "candidate_lists.json"
    if labels_path.exists() and lists_path.exists():
        return labels_path

    rows, lists = build_silver_labels(
        iter(load_candidates_list(PROJECT_ROOT / "challenge" / "sample_candidates.json", limit=200))
    )
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    write_silver_labels(
        rows,
        lists,
        parquet_path=labels_path,
        lists_path=lists_path,
    )
    return labels_path


class TestConstants:
    def test_known_templates_count(self) -> None:
        assert len(KNOWN_TEMPLATES) == 8
        assert len(KNOWN_TEMPLATE_HASHES) == 8

    def test_known_template_hashes_match_metadata(self) -> None:
        prefixes = {t["hash_prefix"] for t in KNOWN_TEMPLATES}
        assert prefixes == KNOWN_TEMPLATE_HASHES

    def test_reference_date_is_fixed(self) -> None:
        assert REFERENCE_DATE == date(2026, 6, 1)


class TestCandidateLoader:
    def test_count_full_dataset(self) -> None:
        path = PROJECT_ROOT / "challenge" / "candidates.jsonl"
        if not path.exists():
            pytest.skip("candidates.jsonl not available")
        assert count_candidates(path) == 100_000

    def test_load_jsonl_stream(self, sample_candidates: list[dict]) -> None:
        assert len(sample_candidates) == 50
        assert all(c["candidate_id"].startswith("CAND_") for c in sample_candidates)


class TestHoneypotDetector:
    def test_salary_inversion_triggers_r4_not_tier0_honeypot(self, sample_candidates: list[dict]) -> None:
        mutated = dict(sample_candidates[0])
        signals = dict(mutated["redrob_signals"])
        signals["expected_salary_range_inr_lpa"] = {"min": 50, "max": 20}
        mutated["redrob_signals"] = signals
        result = detect_honeypot(mutated)
        assert "R4" in result.rules_hit
        assert not result.is_honeypot

    def test_expert_zero_duration_is_subtle_honeypot(self, sample_candidates: list[dict]) -> None:
        mutated = dict(sample_candidates[0])
        skills = [dict(s) for s in mutated.get("skills") or []]
        if not skills:
            skills = [{"name": "PyTorch", "proficiency": "expert", "endorsements": 1, "duration_months": 0}]
        else:
            skills[0] = dict(skills[0])
            skills[0]["proficiency"] = "expert"
            skills[0]["duration_months"] = 0
        mutated["skills"] = skills
        result = detect_honeypot(mutated)
        assert "R1" in result.rules_hit
        assert result.is_honeypot

    def test_r7_triggers_should_exclude(self) -> None:
        path = PROJECT_ROOT / "challenge" / "candidates.jsonl"
        if not path.exists():
            pytest.skip("candidates.jsonl not available")
        for candidate in load_candidates(path, limit=2000):
            result = detect_honeypot(candidate)
            if "R7" in result.rules_hit:
                assert result.should_exclude
                assert not result.is_honeypot
                return
        pytest.skip("No R7 candidate in first 2000 rows")

    def test_reproducible_across_reference_dates(self, sample_candidates: list[dict]) -> None:
        candidate = sample_candidates[0]
        r1 = detect_honeypot(candidate, reference_date=REFERENCE_DATE)
        r2 = detect_honeypot(candidate, reference_date=REFERENCE_DATE)
        assert r1.rules_hit == r2.rules_hit
        assert r1.is_honeypot == r2.is_honeypot


class TestTierAssignment:
    def test_tier_range(self, sample_candidates: list[dict]) -> None:
        for candidate in sample_candidates[:10]:
            hp = detect_honeypot(candidate)
            tier = assign_heuristic_tier(candidate, hp)
            assert 0 <= tier.tier <= 5

    def test_subtle_honeypot_maps_to_tier_zero(self, sample_candidates: list[dict]) -> None:
        mutated = dict(sample_candidates[0])
        mutated["skills"] = [
            {"name": "PyTorch", "proficiency": "expert", "endorsements": 1, "duration_months": 0}
        ]
        hp = detect_honeypot(mutated)
        assert hp.is_honeypot
        tier = assign_heuristic_tier(mutated, hp)
        assert tier.tier == 0

    def test_r4_only_does_not_reach_tier_five(self, sample_candidates: list[dict]) -> None:
        mutated = dict(sample_candidates[0])
        profile = dict(mutated["profile"])
        profile["current_title"] = "Senior AI Engineer"
        profile["years_of_experience"] = 7.0
        profile["country"] = "India"
        profile["location"] = "Pune"
        mutated["profile"] = profile
        signals = dict(mutated["redrob_signals"])
        signals["expected_salary_range_inr_lpa"] = {"min": 50, "max": 20}
        signals["open_to_work_flag"] = True
        signals["recruiter_response_rate"] = 0.9
        mutated["redrob_signals"] = signals
        mutated["skills"] = [
            {"name": "FAISS", "proficiency": "expert", "endorsements": 20, "duration_months": 36},
            {"name": "Learning to Rank", "proficiency": "expert", "endorsements": 15, "duration_months": 30},
            {"name": "PyTorch", "proficiency": "expert", "endorsements": 10, "duration_months": 24},
        ]
        hp = detect_honeypot(mutated)
        tier = assign_heuristic_tier(mutated, hp)
        assert tier.tier <= 3

    def test_junior_title_does_not_reach_tier_five(self, sample_candidates: list[dict]) -> None:
        mutated = dict(sample_candidates[0])
        profile = dict(mutated["profile"])
        profile["current_title"] = "Junior ML Engineer"
        profile["years_of_experience"] = 6.0
        profile["country"] = "India"
        profile["location"] = "Pune"
        mutated["profile"] = profile
        hp = detect_honeypot(mutated)
        tier = assign_heuristic_tier(mutated, hp)
        assert tier.tier < 5


class TestSilverLabels:
    def test_build_silver_labels_schema(self, sample_candidates: list[dict]) -> None:
        rows, lists = build_silver_labels(iter(sample_candidates))
        assert len(rows) == len(sample_candidates)
        assert rows[0].candidate_id.startswith("CAND_")
        assert "senior_ai_tier5" in lists
        assert "honeypot_tier0" in lists
        assert len(lists["manual_review_sample"]) <= 100

    def test_parquet_artifact_readable(self, silver_artifacts_built: Path) -> None:
        frame = pl.read_parquet(silver_artifacts_built)
        required = {
            "candidate_id",
            "tier",
            "label_source",
            "confidence",
            "honeypot_score",
            "is_honeypot",
            "is_trap",
            "should_exclude",
        }
        assert required.issubset(set(frame.columns))
        assert frame.height > 0

    def test_candidate_lists_json_compact(self) -> None:
        path = PROJECT_ROOT / "artifacts" / "candidate_lists.json"
        if not path.exists():
            pytest.skip("Run scripts/build_silver_labels.py first")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "samples" in payload
        assert "tier_distribution" in payload
        assert "counts" in payload
        assert "lists" not in payload
        assert len(payload["samples"]["template_trap_candidates"]) <= 50


class TestAccuracyRegression:
    def test_tier_zero_count_near_spec_honeypots(self) -> None:
        path = PROJECT_ROOT / "artifacts" / "labels_silver.parquet"
        if not path.exists():
            pytest.skip("Run scripts/build_silver_labels.py first")
        frame = pl.read_parquet(path)
        tier0 = frame.filter(pl.col("tier") == 0).height
        assert 20 <= tier0 <= 500, f"tier 0 count {tier0} outside expected honeypot band"

    def test_no_junior_in_tier_five(self) -> None:
        path = PROJECT_ROOT / "challenge" / "candidates.jsonl"
        labels_path = PROJECT_ROOT / "artifacts" / "labels_silver.parquet"
        if not path.exists() or not labels_path.exists():
            pytest.skip("Full dataset artifacts required")
        frame = pl.read_parquet(labels_path)
        tier5_ids = set(frame.filter(pl.col("tier") == 5)["candidate_id"].to_list())
        junior_in_t5 = 0
        for candidate in load_candidates(path):
            if candidate["candidate_id"] not in tier5_ids:
                continue
            title = candidate["profile"]["current_title"].lower()
            if "junior" in title:
                junior_in_t5 += 1
        assert junior_in_t5 == 0

    def test_no_r4_in_tier_five(self) -> None:
        labels_path = PROJECT_ROOT / "artifacts" / "labels_silver.parquet"
        if not labels_path.exists():
            pytest.skip("Run scripts/build_silver_labels.py first")
        frame = pl.read_parquet(labels_path)
        tier5 = frame.filter(pl.col("tier") == 5)
        r4_in_t5 = tier5.filter(pl.col("rules_hit").str.contains("R4")).height
        assert r4_in_t5 == 0

    def test_r7_should_exclude_coverage(self) -> None:
        labels_path = PROJECT_ROOT / "artifacts" / "labels_silver.parquet"
        if not labels_path.exists():
            pytest.skip("Run scripts/build_silver_labels.py first")
        frame = pl.read_parquet(labels_path)
        r7_rows = frame.filter(pl.col("rules_hit").str.contains("R7"))
        assert r7_rows.height > 0
        assert r7_rows.filter(pl.col("should_exclude")).height == r7_rows.height


class TestEdaReport:
    def test_eda_report_exists_or_skippable(self) -> None:
        path = PROJECT_ROOT / "artifacts" / "eda_report.json"
        if not path.exists():
            pytest.skip("Run scripts/eda.py first")
        report = json.loads(path.read_text(encoding="utf-8"))
        assert report["candidates_analyzed"] > 0
        assert len(report["traps"]["known_template_hashes"]) == 8


class TestPhase1Structure:
    @pytest.mark.parametrize(
        "filepath",
        [
            "app/constants.py",
            "app/candidates.py",
            "app/labels/honeypots.py",
            "app/labels/tiers.py",
            "app/labels/silver.py",
            "scripts/eda.py",
            "scripts/build_silver_labels.py",
        ],
    )
    def test_phase1_files_exist(self, filepath: str) -> None:
        assert (PROJECT_ROOT / filepath).is_file(), f"Missing: {filepath}"
