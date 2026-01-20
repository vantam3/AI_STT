import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.stt.manager import SessionManager, StartSessionRequest
from app.config import settings


router = APIRouter()
logging.getLogger("stt")

DEVICE = settings.DEVICE
COMPUTE_TYPE = settings.COMPUTE_TYPE
DEFAULT_MODEL_SIZE = settings.DEFAULT_MODEL_SIZE
MAX_SESSIONS = settings.MAX_SESSIONS
RTP_HOST = settings.RTP_HOST
RTP_PORT_MIN = settings.RTP_PORT_MIN
RTP_PORT_MAX = settings.RTP_PORT_MAX

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
            rtp_host = RTP_HOST
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
