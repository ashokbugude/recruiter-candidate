#!/usr/bin/env python3
"""Package a Hugging Face Docker Space repo with full offline artifacts."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.artifact_names import RANK_OPTIONAL_ARTIFACTS, RANK_REQUIRED_ARTIFACTS  # noqa: E402

# Extra preprocess files validate_artifacts.py expects (not all used at sample rank).
HF_EXTRA_ARTIFACTS: tuple[str, ...] = (
    "gemini_tiers.parquet",
)

COPY_DIRS = ("app", "scripts")
COPY_FILES = (
    "Dockerfile",
    "requirements.txt",
    "pyproject.toml",
    "rank.py",
)
COPY_CHALLENGE = (
    "challenge/candidates.jsonl",
    "challenge/sample_candidates.json",
    "challenge/candidate_schema.json",
    "challenge/validate_submission.py",
)


def _required_artifact_names() -> tuple[str, ...]:
    return RANK_REQUIRED_ARTIFACTS + HF_EXTRA_ARTIFACTS


def validate_local_artifacts(artifacts_dir: Path) -> list[str]:
    missing = [name for name in _required_artifact_names() if not (artifacts_dir / name).exists()]
    return missing


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def package_hf_space(
    out_dir: Path,
    *,
    artifacts_dir: Path,
    skip_validate: bool = False,
) -> None:
    missing = validate_local_artifacts(artifacts_dir)
    if missing and not skip_validate:
        raise FileNotFoundError(
            "Missing rank-ready artifacts:\n  "
            + "\n  ".join(missing)
            + "\n\nRun: python scripts/preprocess.py && python scripts/train_ltr.py"
        )

    candidates_jsonl = PROJECT_ROOT / "challenge" / "candidates.jsonl"
    if not candidates_jsonl.is_file() and not skip_validate:
        raise FileNotFoundError(
            "Missing full candidate pool: challenge/candidates.jsonl\n\n"
            "Unpack candidates.jsonl.gz from the hackathon bundle into challenge/."
        )

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    for name in COPY_FILES:
        shutil.copy2(PROJECT_ROOT / name, out_dir / name)

    for rel in COPY_CHALLENGE:
        dest = out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT / rel, dest)

    for dirname in COPY_DIRS:
        copy_tree(PROJECT_ROOT / dirname, out_dir / dirname)

    copy_tree(artifacts_dir, out_dir / "artifacts")

    shutil.copy2(PROJECT_ROOT / "hf-space" / "README.md", out_dir / "README.md")
    shutil.copy2(PROJECT_ROOT / "hf-space" / ".gitattributes", out_dir / ".gitattributes")
    shutil.copy2(PROJECT_ROOT / "hf-space" / ".dockerignore", out_dir / ".dockerignore")

    print(f"Packaged HF Space -> {out_dir}")
    print(f"  Artifacts: {len(list((out_dir / 'artifacts').iterdir()))} files")
    print("\nNext steps:")
    print("  1. cd", out_dir)
    print("  2. git lfs install")
    print("  3. git init && git branch -M main")
    print("  4. git remote add origin https://huggingface.co/spaces/<user>/redrob-ranker")
    print("  5. git add . && git commit -m 'HF Docker Space'")
    print("  6. git push -u origin main   # HF token as password when prompted")
    print("\nVerify locally: docker build -t redrob-ranker . && docker run -p 7860:7860 redrob-ranker")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build HF Space folder with full artifacts.")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "dist" / "hf-space",
        help="Output directory (push this to HF Space git repo).",
    )
    parser.add_argument("--artifacts", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument("--run-validate-script", action="store_true", help="Also run scripts/validate_artifacts.py")
    args = parser.parse_args()

    artifacts_dir = args.artifacts.resolve()
    try:
        package_hf_space(args.out.resolve(), artifacts_dir=artifacts_dir, skip_validate=args.skip_validate)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.run_validate_script:
        code = subprocess.call(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "validate_artifacts.py"), "--artifacts", str(artifacts_dir)]
        )
        if code != 0:
            return code

    optional_missing = [n for n in RANK_OPTIONAL_ARTIFACTS if not (artifacts_dir / n).exists()]
    if optional_missing:
        print("Optional artifacts not packaged (OK):", ", ".join(optional_missing))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
