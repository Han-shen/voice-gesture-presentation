from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from gesture import run_gesture_detection
from system_controller import SystemController
from voice import listen_commands

app = FastAPI(title="Voice-Gesture Presentation Control v3.0 (Thread-safe)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_base_dir = Path(__file__).resolve().parent
app.state.system = SystemController(config_path=_base_dir / "config.json")


@app.get("/")
async def home():
    return {
        "status": "Voice-Gesture Presentation Control v3.0 (Thread-safe)",
        "endpoints": [
            "/start",
            "/stop",
            "/status",
            "/pause/{module}",
            "/resume/{module}",
            "/logs/stream",
        ],
    }


@app.get("/status")
async def status():
    return app.state.system.get_status()


@app.get("/logs/stream")
async def logs_stream():
    # Server-Sent Events (SSE)
    generator = app.state.system.stream_logs()
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.get("/start")
async def start_system():
    current = app.state.system.get_status()
    if current.get("overall_running"):
        return JSONResponse(status_code=400, content={"error": "Already running"})

    app.state.system.start(gesture_runner=run_gesture_detection, voice_runner=listen_commands)
    return {"message": "System started", "check": "/status"}


@app.get("/stop")
async def stop_system():
    current = app.state.system.get_status()
    if not current.get("overall_running"):
        return JSONResponse(status_code=400, content={"error": "Not running"})

    app.state.system.stop()
    return {"message": "System stopped"}


@app.get("/pause/{module}")
async def pause_module(module: str):
    return app.state.system.pause_module(module=module, paused=True)


@app.get("/resume/{module}")
async def resume_module(module: str):
    return app.state.system.pause_module(module=module, paused=False)


@app.on_event("shutdown")
def on_shutdown():
    try:
        if app.state.system.get_status().get("overall_running"):
            app.state.system.stop()
    except Exception:
        pass

