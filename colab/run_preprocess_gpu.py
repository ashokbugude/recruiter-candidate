#!/usr/bin/env python3
"""Colab helper — run GPU preprocessing with outputs in artifacts/."""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COLAB_ROOT = Path(__file__).resolve().parent


def _run(cmd: list[str], *, cwd: Path) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def check_cuda() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"CUDA OK: {name}")
            return "cuda"
    except ImportError:
        pass
    print("WARNING: CUDA not available — embeddings will run on CPU (slow).")
    return "cpu"


def zip_artifacts(artifacts_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(artifacts_dir.rglob("*")):
            if path.is_file() and not path.name.endswith(".log"):
                zf.write(path, arcname=str(path.relative_to(artifacts_dir.parent)))
    print(f"Created {zip_path} ({zip_path.stat().st_size / 1_048_576:.1f} MB)")


def check_gemini_key() -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    from app.config import get_settings
    from app.gemini_client import get_api_key

    settings = get_settings()
    key = get_api_key(settings)
    if not key:
        raise SystemExit(
            "GEMINI_API_KEY not set. Add Colab Secret GEMINI_API_KEY or set os.environ before running."
        )
    print(f"Gemini API key OK (…{key[-4:]})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Colab GPU preprocess pipeline.")
    parser.add_argument(
        "--step",
        choices=("all", "embeddings", "features", "bm25", "labels"),
        default="all",
        help="Preprocess step (default: all with --skip-llm).",
    )
    parser.add_argument("--skip-llm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-llm", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--zip", type=Path, default=COLAB_ROOT / "artifacts_download.zip")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(PROJECT_ROOT))
    device = check_cuda()

    if not args.skip_llm:
        check_gemini_key()
        print("Gemini enabled: JD parse + 100K archetype labels (~$15–25 Flash, ~1–2 hrs)")
    else:
        print("Gemini skipped (--skip-llm): silver/heuristic labels only")

    candidates = PROJECT_ROOT / "challenge" / "candidates.jsonl"
    if not candidates.exists():
        print(f"Missing {candidates}")
        print("Upload candidates.jsonl — see colab/inputs/README.md")
        return 1

    _run([sys.executable, "scripts/setup_artifacts.py"], cwd=PROJECT_ROOT)

    silver = PROJECT_ROOT / "artifacts" / "labels_silver.parquet"
    if not silver.exists():
        print("Building silver labels (no API)...")
        _run([sys.executable, "scripts/build_silver_labels.py"], cwd=PROJECT_ROOT)

    cmd = [
        sys.executable,
        "scripts/preprocess.py",
        "--step",
        args.step if args.step != "all" else "all",
        "--device",
        device,
    ]
    if args.force:
        cmd.append("--force")
    if args.skip_llm:
        cmd.append("--skip-llm")
    elif args.force or args.force_llm:
        cmd.append("--force-llm")
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])

    _run(cmd, cwd=PROJECT_ROOT)

    artifacts = PROJECT_ROOT / "artifacts"
    if not args.no_zip:
        zip_artifacts(artifacts, args.zip)

    print("\nDone. Copy artifacts/ from Colab to your local repo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
