import os
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse

from app.stt.manager import SessionManager, StartSessionRequest


router = APIRouter()
logging.getLogger("stt")

DEVICE = os.getenv("DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")
DEFAULT_MODEL_SIZE = os.getenv("DEFAULT_MODEL_SIZE", "small")
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "4"))

manager = SessionManager(
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    default_model_size=DEFAULT_MODEL_SIZE,
    max_sessions=MAX_SESSIONS,
)


@router.post("/v1/sessions")
async def start_session(req: StartSessionRequest):
    try:
        session_id = await manager.start(req)
        rtp_host = None
        rtp_port = None
        payload_type = None
        if req.input.type == "janus_rtp_opus":
            rtp_host = req.input.listen_ip
            rtp_port = req.input.audio_port
            payload_type = req.input.payload_type
        return JSONResponse(
            {
                "session_id": session_id,
                "rtp_host": rtp_host,
                "rtp_port": rtp_port,
                "payload_type": payload_type,
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/v1/sessions/{session_id}")
async def stop_session(session_id: str):
    ok = await manager.stop(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="session not found")
    return {"ok": True}


@router.get("/v1/sessions")
async def list_sessions():
    return manager.list()


@router.websocket("/v1/ws/{session_id}")
async def ws_captions(ws: WebSocket, session_id: str):
    await ws.accept()
    queue = manager.subscribe(session_id)
    if queue is None:
        await ws.close(code=1008)
        return
    try:
        while True:
            msg = await queue.get()
            await ws.send_json(msg)
    except WebSocketDisconnect:
        pass
    finally:
        manager.unsubscribe(session_id, queue)
