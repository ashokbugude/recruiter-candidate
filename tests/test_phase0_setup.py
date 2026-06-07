"""Phase 0 setup verification tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import PROJECT_ROOT, Settings, get_settings

REQUIRED_DIRS = ["app", "scripts", "artifacts", "challenge", "tests", "prompts", "docs"]
REQUIRED_FILES = [
    "rank.py",
    "requirements.txt",
    "pyproject.toml",
    "README.md",
    "submission_metadata.yaml",
    ".env.example",
    ".gitignore",
    "app/config.py",
    "app/logging_setup.py",
    "scripts/setup_artifacts.py",
    "challenge/candidates.jsonl",
    "challenge/validate_submission.py",
]


@pytest.fixture
def root() -> Path:
    return PROJECT_ROOT


class TestProjectStructure:
    @pytest.mark.parametrize("dirname", REQUIRED_DIRS)
    def test_required_directories_exist(self, root: Path, dirname: str) -> None:
        assert (root / dirname).is_dir(), f"Missing directory: {dirname}"

    @pytest.mark.parametrize("filepath", REQUIRED_FILES)
    def test_required_files_exist(self, root: Path, filepath: str) -> None:
        assert (root / filepath).is_file(), f"Missing file: {filepath}"


class TestConfiguration:
    def test_settings_defaults(self) -> None:
        settings = Settings()
        assert settings.top_k_output == 100
        assert settings.max_runtime_seconds == 300
        assert settings.max_ram_gb == 16
        assert settings.recall_pool_size == 2000
        assert settings.rerank_pool_size == 400

    def test_settings_singleton(self) -> None:
        get_settings.cache_clear()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_artifact_path_resolution(self) -> None:
        settings = Settings()
        path = settings.artifact_path("jd_requirements.json")
        assert path.parent == settings.artifacts_dir.resolve()


class TestArtifacts:
    def test_job_description_exists_after_setup(self, root: Path) -> None:
        jd = root / "artifacts" / "job_description.txt"
        assert jd.exists(), "Run: python scripts/setup_artifacts.py"
        assert jd.stat().st_size > 500
        content = jd.read_text(encoding="utf-8")
        assert "Senior AI Engineer" in content

    def test_submission_spec_exists(self, root: Path) -> None:
        spec = root / "artifacts" / "submission_spec.txt"
        assert spec.exists()
        assert "NDCG@10" in spec.read_text(encoding="utf-8")


class TestRankCLI:
    def test_rank_help_exits_zero(self) -> None:
        import subprocess

        result = subprocess.run(
            ["python", "rank.py", "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--candidates" in result.stdout

    def test_rank_validates_missing_jd_when_artifacts_empty(self, tmp_path: Path) -> None:
        """Ranker should fail clearly if artifacts were never bootstrapped."""
        import subprocess

        empty_artifacts = tmp_path / "empty_artifacts"
        empty_artifacts.mkdir()
        # Use sample file to avoid loading 100K
        candidates = PROJECT_ROOT / "challenge" / "sample_candidates.json"
        if not candidates.exists():
            pytest.skip("sample_candidates.json not available")

        result = subprocess.run(
            [
                "python",
                "rank.py",
                "--candidates",
                str(candidates),
                "--out",
                str(tmp_path / "out.csv"),
                "--artifacts",
                str(empty_artifacts),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        # Without job_description in custom empty artifacts dir → error
        assert result.returncode == 1
