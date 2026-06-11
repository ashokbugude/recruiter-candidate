"""Submission CSV format (submission_spec §2)."""

from app.submission_csv import format_submission_csv


def test_format_submission_csv_columns_and_header() -> None:
    rows = [
        {
            "candidate_id": "CAND_0000001",
            "rank": 1,
            "score": 0.99,
            "reasoning": "Strong fit for production search.",
        }
    ]
    csv_text = format_submission_csv(rows)
    lines = csv_text.strip().splitlines()
    assert lines[0] == "candidate_id,rank,score,reasoning"
    assert "CAND_0000001,1,0.99" in lines[1]
