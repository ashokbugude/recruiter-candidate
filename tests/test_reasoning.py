"""Reasoning string tests."""

from __future__ import annotations

import json

from app.config import PROJECT_ROOT
from app.jd_requirements import build_heuristic_requirements
from app.reasoning import build_reasoning


def test_reasoning_varies_by_rank_band() -> None:
    path = PROJECT_ROOT / "challenge" / "sample_candidates.json"
    candidates = json.loads(path.read_text(encoding="utf-8"))
    jd = build_heuristic_requirements(
        (PROJECT_ROOT / "artifacts" / "job_description.txt").read_text(encoding="utf-8")
    )
    top = build_reasoning(candidates[0], rank=1, jd=jd)
    mid = build_reasoning(candidates[0], rank=40, jd=jd)
    tail = build_reasoning(candidates[0], rank=90, jd=jd)
    assert top.startswith("Top pick:")
    assert "Pune/Noida" in top or "Pune/Noida" in mid
    assert "Solid bench:" in mid or "Reserve:" in mid or "Strong match:" in mid
    assert any(x in tail for x in ("Depth option", "Pipeline backup", "Extended shortlist", "Reserve:"))
    assert len(top) <= 500
