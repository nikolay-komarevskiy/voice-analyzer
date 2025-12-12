from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import Settings, load_settings
from .pipeline import AudioBroadcaster, AudioStreamManager

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config.yaml"

settings = load_settings(CONFIG_PATH)
stream_manager = AudioStreamManager(settings)
broadcaster = AudioBroadcaster(stream_manager, settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await stream_manager.start()
    await broadcaster.start()
    try:
        yield
    finally:
        await broadcaster.stop()
        await stream_manager.stop()


app = FastAPI(title="Voice Analyzer", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/api/audio-config")
async def get_config() -> Dict[str, Any]:
    return settings.to_dict()


@app.post("/api/audio-config")
async def update_config(payload: Dict[str, Any]):
    global settings
    settings = settings.merge(payload)
    await stream_manager.restart(settings)
    await broadcaster.restart(settings)
    return settings.to_dict()


@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        async for message in broadcaster.stream():
            await websocket.send_json(message)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as exc:
        logger.exception("WebSocket error: %s", exc)
        await websocket.close(code=1011, reason=str(exc))


static_dir = BASE_DIR
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
