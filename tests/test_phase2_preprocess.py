"""Phase 2 offline preprocessing verification tests."""

from __future__ import annotations

import json

import polars as pl
import pytest

from app.bm25_index import build_bm25_index, load_bm25, search_bm25
from app.candidates import load_candidates_list
from app.config import PROJECT_ROOT, Settings
from app.feature_store import build_features_frame, validate_features_frame
from app.features import FEATURE_NAMES, extract_features
from app.jd_requirements import build_heuristic_requirements, load_or_build_jd_requirements


@pytest.fixture(scope="module")
def jd_requirements(tmp_path_factory) -> object:
    jd_path = PROJECT_ROOT / "artifacts" / "job_description.txt"
    if not jd_path.exists():
        pytest.skip("Run scripts/setup_artifacts.py first")
    out = tmp_path_factory.mktemp("jd") / "jd_requirements.json"
    settings = Settings(jd_requirements_path=out, job_description_path=jd_path)
    return load_or_build_jd_requirements(jd_path, out, settings=settings, force=True)


@pytest.fixture(scope="module")
def sample_candidates() -> list[dict]:
    path = PROJECT_ROOT / "challenge" / "sample_candidates.json"
    if not path.exists():
        pytest.skip("sample_candidates.json not available")
    return load_candidates_list(path, limit=30)


class TestJDRequirements:
    def test_heuristic_requirements_schema(self) -> None:
        jd_text = (PROJECT_ROOT / "artifacts" / "job_description.txt").read_text(encoding="utf-8")
        req = build_heuristic_requirements(jd_text)
        assert req.role_title
        assert len(req.must_have_skills) >= 5
        assert req.yoe_min == 5.0

    def test_cached_jd_requirements_file(self, jd_requirements) -> None:
        assert len(jd_requirements.must_have_keywords) > 10


class TestFeatureExtraction:
    def test_feature_count(self, sample_candidates, jd_requirements) -> None:
        feats = extract_features(sample_candidates[0], jd_requirements)
        assert len(feats) == 45
        assert set(feats.keys()) == set(FEATURE_NAMES)

    def test_features_frame_validation(self, sample_candidates, jd_requirements) -> None:
        frame = build_features_frame(iter(sample_candidates), jd_requirements)
        validate_features_frame(frame)
        assert frame.height == len(sample_candidates)


class TestBM25:
    def test_build_and_search(self, sample_candidates) -> None:
        artifacts = build_bm25_index(sample_candidates)
        assert artifacts.corpus_size == len(sample_candidates)
        results = search_bm25(artifacts, "retrieval ranking embeddings python", top_k=5)
        assert len(results) >= 1
        assert results[0][0].startswith("CAND_")


class TestPhase2Artifacts:
    @pytest.mark.parametrize(
        "filepath",
        [
            "app/features.py",
            "app/feature_store.py",
            "app/embeddings.py",
            "app/bm25_index.py",
            "app/gemini_client.py",
            "app/jd_requirements.py",
            "prompts/parse_jd.txt",
            "prompts/label_archetype.txt",
            "scripts/parse_jd.py",
            "scripts/label_archetypes.py",
            "scripts/preprocess.py",
        ],
    )
    def test_phase2_files_exist(self, filepath: str) -> None:
        assert (PROJECT_ROOT / filepath).is_file(), f"Missing: {filepath}"

    def test_preprocess_sample_artifacts(self, tmp_path, sample_candidates, jd_requirements) -> None:
        """Build features + BM25 on sample without downloading BGE."""
        from app.bm25_index import save_bm25
        from app.feature_store import write_features_parquet

        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        frame = build_features_frame(iter(sample_candidates), jd_requirements)
        write_features_parquet(frame, artifacts / "candidate_features.parquet")

        bm25 = build_bm25_index(sample_candidates)
        save_bm25(bm25, artifacts / "bm25.pkl")
        loaded = load_bm25(artifacts / "bm25.pkl")
        assert loaded.corpus_size == len(sample_candidates)

        written = pl.read_parquet(artifacts / "candidate_features.parquet")
        assert written.height == len(sample_candidates)


class TestParseJDScript:
    def test_parse_jd_creates_json(self, tmp_path) -> None:
        import subprocess

        jd = PROJECT_ROOT / "artifacts" / "job_description.txt"
        if not jd.exists():
            pytest.skip("JD not bootstrapped")
        out = tmp_path / "jd_requirements.json"
        result = subprocess.run(
            ["python", "scripts/parse_jd.py", "--out", str(out)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert "must_have_skills" in payload
