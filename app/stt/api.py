import os
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.stt.manager import SessionManager, StartSessionRequest


router = APIRouter()
logging.getLogger("stt")

DEVICE = os.getenv("DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")
DEFAULT_MODEL_SIZE = os.getenv("DEFAULT_MODEL_SIZE", "small")
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "4"))
RTP_HOST = os.getenv("RTP_HOST")
RTP_PORT_MIN = int(os.getenv("RTP_PORT_MIN", "40000"))
RTP_PORT_MAX = int(os.getenv("RTP_PORT_MAX", "49999"))

manager = SessionManager(
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    default_model_size=DEFAULT_MODEL_SIZE,
    max_sessions=MAX_SESSIONS,
)


@router.post("/v1/sessions")
async def start_session(req: StartSessionRequest):
    try:
        if RTP_PORT_MIN < 1 or RTP_PORT_MAX > 65535 or RTP_PORT_MIN > RTP_PORT_MAX:
            raise ValueError("RTP_PORT_MIN/MAX must be within 1-65535 and MIN <= MAX")
        session_id = await manager.start(req, rtp_port_range=(RTP_PORT_MIN, RTP_PORT_MAX))
        rtp_host = None
        rtp_port = None
        payload_type = None
        if req.input.type == "janus_rtp_opus":
            rtp_host = RTP_HOST or "10.103.100.233"
            rtp_port = req.input.audio_port
            payload_type = 111
            logging.getLogger("stt").info(
                "session %s rtp config host=%s port=%s pt=%s model=%s",
                session_id,
                rtp_host,
                rtp_port,
                payload_type,
                DEFAULT_MODEL_SIZE,
            )
        return JSONResponse(
            {
                "session_id": session_id,
                "rtp_host": rtp_host,
                "rtp_port": rtp_port,
                "payload_type": payload_type,
                "type": "audio",
                "codec": "opus",
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


@router.get("/v1/sessions/{session_id}/transcript")
async def get_transcript(session_id: str):
    transcript = manager.get_transcript(session_id)
    if transcript is None:
        raise HTTPException(status_code=404, detail="session not found")
    return transcript

