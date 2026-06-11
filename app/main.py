"""FastAPI sandbox — bundled candidates.jsonl, live rank.py, portal CSV download."""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from app.artifacts_check import ArtifactValidationError, validate_artifacts
from app.config import get_settings
from app.logging_setup import configure_logging
from app.rank_submission import run_portal_submission_rank
from app.submission_csv import format_submission_csv

logger = logging.getLogger(__name__)

UPLOAD_MAX_CANDIDATES = 100_000
SUBMISSION_FILENAME = "team_sarva_automata.csv"
BUNDLED_POOL_NAME = "candidates.jsonl"

_artifacts_ok = False
_artifact_warnings: list[str] = []
_bundled_pool_count: int = 0


def _count_jsonl_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _artifacts_ok, _artifact_warnings, _bundled_pool_count
    settings = get_settings()
    configure_logging(settings.log_level)
    try:
        _artifact_warnings = validate_artifacts(settings.artifacts_dir.resolve())
        for msg in _artifact_warnings:
            logger.warning("%s", msg)
        candidates_path = settings.candidates_path.resolve()
        if not candidates_path.is_file():
            raise ArtifactValidationError(f"Bundled candidate pool missing: {candidates_path}")
        _bundled_pool_count = _count_jsonl_lines(candidates_path)
        _artifacts_ok = True
        logger.info("Bundled pool ready: %s (%d candidates)", candidates_path, _bundled_pool_count)
    except ArtifactValidationError as exc:
        logger.error("Artifact validation failed: %s", exc)
        _artifacts_ok = False
        _bundled_pool_count = 0
    yield


app = FastAPI(
    title="Redrob Candidate Ranker",
    description=(
        "Sandbox per submission_spec §10.5: pre-loaded candidates.jsonl in Docker, "
        "ranked CSV download → team_sarva_automata.csv."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class RankRequest(BaseModel):
    candidates_path: str | None = Field(default=None, description="Server path to candidates.jsonl")
    top_k: int = Field(default=100, ge=1, le=100)


def _require_artifacts() -> None:
    if not _artifacts_ok:
        raise HTTPException(status_code=503, detail="Artifacts missing or invalid")


def _csv_response(
    rows: list[dict],
    filename: str,
    *,
    duration_seconds: float | None = None,
) -> Response:
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    if duration_seconds is not None:
        headers["X-Rank-Duration-Seconds"] = f"{duration_seconds:.2f}"
    return Response(
        content=format_submission_csv(rows),
        media_type="text/csv; charset=utf-8",
        headers=headers,
    )


def _parse_upload_lines(raw: bytes) -> list[str]:
    text = raw.decode("utf-8").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty upload")

    if text.startswith("["):
        try:
            records = json.loads(text)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON array: {exc}") from exc
        if not isinstance(records, list):
            raise HTTPException(status_code=400, detail="JSON upload must be an array of candidates")
        return [json.dumps(row) for row in records if row]

    return [ln for ln in text.splitlines() if ln.strip()]


def _validate_upload_records(lines: list[str]) -> None:
    for index, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON on line {index}: {exc}") from exc
        if not isinstance(row, dict) or not row.get("candidate_id"):
            raise HTTPException(status_code=400, detail=f"Missing candidate_id on line {index}")


def _write_jsonl_temp(lines: list[str]) -> Path:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as tmp:
        tmp.write("\n".join(lines))
        return Path(tmp.name)


def _run_portal_submission_rank(candidates_path: Path | None = None) -> tuple[list[dict], float]:
    """Blocking rank via rank.py (bundled pool or uploaded file)."""
    settings = get_settings()
    path = (candidates_path or settings.candidates_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing candidates pool: {path}")
    started = time.perf_counter()
    results = run_portal_submission_rank(
        artifacts_dir=settings.artifacts_dir.resolve(),
        settings=settings,
        candidates_path=path,
    )
    duration = time.perf_counter() - started
    logger.info("Rank finished in %.2fs on %s (%d output rows)", duration, path.name, len(results))
    return results, duration


@app.get("/pool")
def pool_info() -> dict:
    """Bundled candidate pool metadata (pre-loaded in Docker image)."""
    _require_artifacts()
    settings = get_settings()
    path = settings.candidates_path.resolve()
    return {
        "bundled": True,
        "file": BUNDLED_POOL_NAME,
        "path": str(path),
        "candidate_count": _bundled_pool_count,
        "ready": True,
    }


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    pool_label = (
        f"{BUNDLED_POOL_NAME} ({_bundled_pool_count:,} candidates)"
        if _artifacts_ok
        else f"{BUNDLED_POOL_NAME} (not loaded)"
    )
    status = "ready" if _artifacts_ok else "artifacts unavailable — check /health"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Redrob Candidate Ranker — Sandbox</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 44rem; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }}
    h2 {{ margin: 0 0 0.5rem; font-size: 1.05rem; }}
    a {{ color: #2563eb; }}
    button {{ cursor: pointer; padding: 0.55rem 1rem; font-size: 1rem; border-radius: 6px; border: 1px solid transparent; }}
    button:disabled {{ opacity: 0.6; cursor: wait; }}
    code {{ background: #f3f4f6; padding: 0.1rem 0.35rem; border-radius: 4px; }}
    .row {{ display: flex; align-items: center; gap: 0.75rem; flex-wrap: wrap; margin-top: 0.75rem; }}
    .loader {{ display: none; align-items: center; gap: 0.4rem; color: #374151; font-size: 0.95rem; }}
    .loader.is-active {{ display: inline-flex; }}
    .muted {{ color: #6b7280; font-size: 0.9rem; }}
    .panel {{ border-radius: 10px; padding: 1rem 1.1rem; margin: 1rem 0; }}
    .panel-default {{ background: #f0fdf4; border: 2px solid #86efac; }}
    .panel-default h2 {{ color: #166534; }}
    .panel-default button {{ background: #16a34a; color: #fff; border-color: #15803d; }}
    .panel-default button:hover:not(:disabled) {{ background: #15803d; }}
    .panel-upload {{ background: #f8fafc; border: 2px dashed #94a3b8; }}
    .panel-upload h2 {{ color: #334155; }}
    .panel-upload button {{ background: #fff; color: #1d4ed8; border-color: #93c5fd; }}
    .panel-upload button:hover:not(:disabled) {{ background: #eff6ff; }}
    .panel-upload button:disabled {{ background: #f1f5f9; color: #64748b; }}
    .badge {{ display: inline-block; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.03em; text-transform: uppercase;
              padding: 0.15rem 0.45rem; border-radius: 4px; margin-bottom: 0.35rem; }}
    .badge-default {{ background: #dcfce7; color: #166534; }}
    .badge-upload {{ background: #e2e8f0; color: #475569; }}
    .status-msg {{ margin: 0.75rem 0 0; min-height: 1.25rem; font-size: 0.95rem; }}
    .status-msg.ok {{ color: #166534; }}
    .status-msg.err {{ color: #b91c1c; }}
  </style>
</head>
<body>
  <h1>Redrob Candidate Ranker</h1>
  <p>Sandbox per <strong>submission_spec §10.5</strong>: rank the pre-loaded candidate pool or your own file,
     then download <code>{SUBMISSION_FILENAME}</code>.</p>
  <p><strong>Status:</strong> {status}</p>

  <section class="panel panel-default" aria-labelledby="default-heading">
    <span class="badge badge-default">Default</span>
    <h2 id="default-heading">Pre-loaded candidate pool</h2>
    <p class="muted">Bundled in the Docker image — no upload needed.</p>
    <p><strong id="pool-file">{pool_label}</strong></p>
    <p class="muted">Estimated time: 4–7 minutes for 100,000 candidates. Check the Space <strong>Logs</strong> tab on Hugging Face for progress while ranking.</p>
    <div class="row">
      <button type="button" id="rank-default-btn">Rank bundled pool &amp; download CSV</button>
      <span id="loader-default" class="loader" aria-live="polite">
        <span aria-hidden="true">⏳</span> <span id="loader-default-text">Ranking…</span>
      </span>
    </div>
    <p id="status-default" class="status-msg" role="status"></p>
  </section>

  <section class="panel panel-upload" aria-labelledby="upload-heading">
    <span class="badge badge-upload">Optional</span>
    <h2 id="upload-heading">Upload your own candidates</h2>
    <p class="muted">JSONL (one JSON per line) or JSON array. Max 100,000 rows.</p>
    <p><input type="file" id="file" name="file" accept=".jsonl,.json,.txt"></p>
    <p class="muted" id="file-hint">Choose a file to enable ranking below. Time scales with row count (4–7 min per 100,000 records).</p>
    <div class="row">
      <button type="button" id="rank-upload-btn" disabled>Rank uploaded file &amp; download CSV</button>
      <span id="loader-upload" class="loader" aria-live="polite">
        <span aria-hidden="true">⏳</span> <span id="loader-upload-text">Ranking…</span>
      </span>
    </div>
    <p id="status-upload" class="status-msg" role="status"></p>
  </section>

  <p><a href="/health">/health</a> · <a href="/pool">/pool</a> · <a href="/docs">API docs</a></p>
  <script>
    const BASE_CANDIDATE_COUNT = 100000;
    const EST_MIN_MINUTES = 4;
    const EST_MAX_MINUTES = 7;
    const LOGS_HINT = " Check the Space Logs tab on Hugging Face for progress updates.";

    function formatDuration(totalSeconds) {{
      const s = Math.max(0, Math.round(Number(totalSeconds) || 0));
      const m = Math.floor(s / 60);
      const rem = s % 60;
      if (m > 0) return `${{m}}m ${{rem}}s`;
      return `${{rem}}s`;
    }}

    function estimateTimeRange(count) {{
      const n = Math.max(1, Number(count) || 1);
      const minSec = Math.max(1, Math.floor((EST_MIN_MINUTES * 60 * n) / BASE_CANDIDATE_COUNT));
      const maxSec = Math.max(minSec + 1, Math.ceil((EST_MAX_MINUTES * 60 * n) / BASE_CANDIDATE_COUNT));
      if (maxSec < 60) {{
        return `typically ${{minSec}}–${{maxSec}} seconds`;
      }}
      const minM = Math.max(1, Math.floor(minSec / 60));
      const maxM = Math.max(minM, Math.ceil(maxSec / 60));
      if (minM === maxM) {{
        return `typically ${{minM}} minute${{minM === 1 ? "" : "s"}}`;
      }}
      return `typically ${{minM}}–${{maxM}} minutes`;
    }}

    async function countCandidates(file) {{
      const text = await file.text();
      const trimmed = text.trim();
      if (!trimmed) return 0;
      if (trimmed.startsWith("[")) {{
        try {{
          const records = JSON.parse(trimmed);
          return Array.isArray(records) ? records.length : 0;
        }} catch (_) {{
          return 0;
        }}
      }}
      return trimmed.split("\\n").filter((line) => line.trim()).length;
    }}

    function setLoading(loader, loaderText, active, elapsedSeconds) {{
      if (active) {{
        loader.classList.add("is-active");
        loaderText.textContent = `Ranking… ${{formatDuration(elapsedSeconds)}}`;
      }} else {{
        loader.classList.remove("is-active");
        loaderText.textContent = "Ranking…";
      }}
    }}

    function startElapsedTimer(loader, loaderText, startedAt) {{
      const tick = () => {{
        const elapsed = (performance.now() - startedAt) / 1000;
        loaderText.textContent = `Ranking… ${{formatDuration(elapsed)}}`;
      }};
      tick();
      return window.setInterval(tick, 1000);
    }}

    async function refreshPool() {{
      try {{
        const r = await fetch("/pool");
        if (!r.ok) return;
        const data = await r.json();
        document.getElementById("pool-file").textContent =
          `${{data.file}} (${{data.candidate_count.toLocaleString()}} candidates)`;
      }} catch (_) {{}}
    }}
    refreshPool();

    const fileInput = document.getElementById("file");
    const uploadBtn = document.getElementById("rank-upload-btn");
    const fileHint = document.getElementById("file-hint");
    let uploadRecordCount = 0;

    fileInput.addEventListener("change", async (event) => {{
      const input = event.target;
      if (input.files && input.files.length) {{
        const file = input.files[0];
        uploadBtn.disabled = true;
        fileHint.textContent = `Counting records in ${{file.name}}…`;
        uploadRecordCount = await countCandidates(file);
        uploadBtn.disabled = uploadRecordCount > 0;
        if (uploadRecordCount > 0) {{
          const est = estimateTimeRange(uploadRecordCount);
          fileHint.textContent =
            `Selected: ${{file.name}} (${{uploadRecordCount.toLocaleString()}} candidates) — ${{est}}.`;
        }} else {{
          fileHint.textContent = "Could not read candidate records from this file.";
        }}
      }} else {{
        uploadBtn.disabled = true;
        uploadRecordCount = 0;
        fileHint.textContent =
          "Choose a file to enable ranking below. Time scales with row count (4–7 min per 100,000 records).";
      }}
    }});

    async function runRank({{ btn, loader, loaderText, statusEl, fetchFn, busyLabel }}) {{
      const started = performance.now();
      let timerId = null;
      btn.disabled = true;
      statusEl.textContent = busyLabel;
      statusEl.className = "status-msg";
      setLoading(loader, loaderText, true, 0);
      timerId = startElapsedTimer(loader, loaderText, started);

      try {{
        const response = await fetchFn();
        if (!response.ok) {{
          let detail = response.statusText;
          try {{
            const payload = await response.json();
            detail = payload.detail || detail;
            if (Array.isArray(detail)) detail = detail.map((d) => d.msg || d).join("; ");
          }} catch (_) {{}}
          throw new Error(detail);
        }}

        const serverSeconds = response.headers.get("X-Rank-Duration-Seconds");
        const elapsedSec = serverSeconds
          ? Number(serverSeconds)
          : (performance.now() - started) / 1000;

        if (timerId !== null) window.clearInterval(timerId);
        setLoading(loader, loaderText, false, 0);
        btn.disabled = false;

        const disposition = response.headers.get("Content-Disposition") || "";
        const match = disposition.match(/filename=\"?([^\";]+)\"?/);
        const filename = match ? match[1] : "{SUBMISSION_FILENAME}";

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);

        statusEl.textContent =
          `Complete in ${{formatDuration(elapsedSec)}}. Download started: ${{filename}}.`;
        statusEl.className = "status-msg ok";
      }} catch (error) {{
        if (timerId !== null) window.clearInterval(timerId);
        setLoading(loader, loaderText, false, 0);
        btn.disabled = (btn === uploadBtn) ? !(fileInput.files && fileInput.files.length) : false;
        statusEl.textContent = "Error: " + (error.message || error);
        statusEl.className = "status-msg err";
      }}
    }}

    document.getElementById("rank-default-btn").addEventListener("click", () => {{
      runRank({{
        btn: document.getElementById("rank-default-btn"),
        loader: document.getElementById("loader-default"),
        loaderText: document.getElementById("loader-default-text"),
        statusEl: document.getElementById("status-default"),
        busyLabel:
          "Ranking bundled pool (100,000 candidates) — typically 4–7 minutes." + LOGS_HINT,
        fetchFn: () => fetch("/rank/run", {{ method: "POST" }}),
      }});
    }});

    uploadBtn.addEventListener("click", () => {{
      if (!fileInput.files || !fileInput.files.length) return;
      const name = fileInput.files[0].name;
      const body = new FormData();
      body.append("file", fileInput.files[0]);
      runRank({{
        btn: uploadBtn,
        loader: document.getElementById("loader-upload"),
        loaderText: document.getElementById("loader-upload-text"),
        statusEl: document.getElementById("status-upload"),
        busyLabel:
          `Ranking ${{uploadRecordCount.toLocaleString()}} candidates from ${{name}} — `
          + `${{estimateTimeRange(uploadRecordCount)}}.` + LOGS_HINT,
        fetchFn: () => fetch("/rank/upload", {{ method: "POST", body }}),
      }});
    }});
  </script>
</body>
</html>"""


@app.get("/health")
def health() -> dict:
    _require_artifacts()
    settings = get_settings()
    return {
        "status": "ok",
        "warnings": _artifact_warnings,
        "bundled_pool": BUNDLED_POOL_NAME,
        "candidate_count": _bundled_pool_count,
        "candidates_path": str(settings.candidates_path.resolve()),
    }


@app.post("/rank/run")
async def rank_run() -> Response:
    """Run rank.py on bundled candidates.jsonl and return portal CSV."""
    _require_artifacts()
    logger.info("Starting rank.py on bundled pool (%d candidates)", _bundled_pool_count)
    try:
        results, duration = await asyncio.to_thread(_run_portal_submission_rank)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _csv_response(results, SUBMISSION_FILENAME, duration_seconds=duration)


@app.post("/rank")
async def rank_candidates(body: RankRequest | None = None) -> Response:
    """Alias for /rank/run (bundled pool only)."""
    if body and body.candidates_path:
        raise HTTPException(
            status_code=400,
            detail="Sandbox uses bundled candidates.jsonl only; call POST /rank/run",
        )
    return await rank_run()


@app.post("/rank/upload")
async def rank_upload(file: UploadFile = File(...)) -> Response:
    """Rank an uploaded JSONL / JSON array (overrides bundled pool)."""
    _require_artifacts()
    lines = _parse_upload_lines(await file.read())
    if not lines:
        raise HTTPException(status_code=400, detail="No candidate records in upload")
    if len(lines) > UPLOAD_MAX_CANDIDATES:
        raise HTTPException(
            status_code=400,
            detail=f"Upload limit: {UPLOAD_MAX_CANDIDATES} candidates (got {len(lines)})",
        )
    _validate_upload_records(lines)

    tmp_path = _write_jsonl_temp(lines)
    try:
        logger.info("Ranking uploaded file %s (%d candidates)", file.filename, len(lines))
        results, duration = await asyncio.to_thread(_run_portal_submission_rank, tmp_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    stem = Path(file.filename or "upload").stem or "upload"
    out_name = SUBMISSION_FILENAME if len(lines) > 100 else f"{stem}_ranked.csv"
    return _csv_response(results, out_name, duration_seconds=duration)


@app.get("/rank/sample")
async def rank_sample() -> Response:
    """API only: run rank.py on bundled pool."""
    return await rank_run()
