"""FastAPI HTTP surface for video-analyser (lens family contract)."""
from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from lens_contract import add_contract_routes, add_cors, upload_tempfile

from video_analyser.core.pipeline_coordinator import PipelineCoordinator
from video_analyser.manifest import MANIFEST
from video_analyser.models import VideoAnalysis

app = FastAPI(title="video-analyser", version=MANIFEST["version"])

# GET /health and GET /manifest (the family contract, via lens-contract).
add_contract_routes(app, MANIFEST)
# CORS — env-driven: VIDEO_ANALYSER_MODE=desktop (Electron) or VIDEO_ANALYSER_ALLOWED_ORIGINS.
add_cors(app, env_prefix="VIDEO_ANALYSER")

_coordinator = PipelineCoordinator()


@app.post("/analyse", response_model=VideoAnalysis)
async def analyse(file: UploadFile = File(...)) -> VideoAnalysis:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=422, detail="Empty file")

    # Video analysis is a minutes-long, CPU/ML-heavy pipeline. Run it in a
    # threadpool so the event loop (and GET /health) stays responsive during the
    # run — that's how a caller/CI confirms the server is still alive.
    with upload_tempfile(content, file.filename) as tmp_path:
        try:
            return await run_in_threadpool(_coordinator.analyse, tmp_path)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc
