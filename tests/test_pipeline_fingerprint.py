"""Lightweight pipeline smoke test on sample candidates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import PROJECT_ROOT


@pytest.mark.skipif(
    not (PROJECT_ROOT / "artifacts" / "ltr_model.lgb").exists(),
    reason="LTR model not built",
)
def test_sample_pipeline_returns_100_rows() -> None:
    from app.config import get_settings
    from app.fusion import settings_with_fusion_params
    from app.pipeline import RankingPipeline

    sample_path = PROJECT_ROOT / "challenge" / "sample_candidates.json"
    if not sample_path.exists():
        pytest.skip("sample_candidates.json missing")

    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as tmp:
        for row in json.loads(sample_path.read_text(encoding="utf-8")):
            tmp.write(json.dumps(row) + "\n")
        tmp_path = Path(tmp.name)

    settings = settings_with_fusion_params(get_settings(), get_settings().artifacts_dir)
    pipeline = RankingPipeline(settings, settings.artifacts_dir)
    try:
        results = pipeline.rank(tmp_path, top_k=min(100, 50))
    finally:
        tmp_path.unlink(missing_ok=True)

    assert len(results) >= 1
    assert results[0]["rank"] == 1
    assert "candidate_id" in results[0]
