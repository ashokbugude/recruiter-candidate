# Track 1 — Portal Upload Guide (Sarva Automata)

## What to upload

| # | Track 1 requirement | Your file |
|---|----------------------|-----------|
| 1 | Working GitHub repository | https://github.com/ashokbugude/recruiter-candidate |
| 2 | Ranked candidate output (provided format) | **`team_sarva_automata.csv`** |
| 3 | Methodology document | **`docs/Sarva_Automata_Methodology.pdf`** |

## Portal form fields

Copy from [`submission_metadata.yaml`](../submission_metadata.yaml):

- Team: **Sarva Automata**
- Primary contact: Arjun Mangarath, arjun00102@gmail.com, +91-8139825799
- Sandbox: https://huggingface.co/spaces/ashokbugude/redrob-ranker
- Reproduce: `python rank.py --candidates ./challenge/candidates.jsonl --out ./team_sarva_automata.csv`
- AI tools: Cursor, Gemini (offline only)

## Pre-upload checks

```bash
python challenge/validate_submission.py team_sarva_automata.csv
python scripts/verify_submission.py --fingerprint
```

## Refresh methodology PDF

```bash
python scripts/build_methodology_deck.py --export-pdf docs/Sarva_Automata_Methodology.pdf
```

## Do not upload

- `sample_ranked.csv` — HF sandbox demo only (24 rows from `sample_candidates.json`)
- `artifacts/` binaries — stay local / HF Space, not main GitHub
