# Prompt Templates

Gemini prompts for **offline preprocessing only**. Not invoked during `rank.py`.

| File | Phase | Purpose |
|------|-------|---------|
| `parse_jd.txt` | 2 | Extract structured requirements from job description |
| `label_archetype.txt` | 2 | Batch relevance tier labeling (0–5) per candidate |

Prompts are used by `scripts/parse_jd.py` and `scripts/label_archetypes.py` (offline only).
