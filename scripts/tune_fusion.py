#!/usr/bin/env python3
"""DEPRECATED — use scripts/tune_modifiers.py for full fusion + modifier tuning."""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "tune_fusion.py is deprecated.\n"
        "Use: python scripts/tune_modifiers.py --trials 120\n"
        "That script tunes fusion weights, modifiers, career RRF weight, and verifies live pipeline.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
