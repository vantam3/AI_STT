import os
import logging
import numpy as np
from fastapi import FastAPI

from app.stt.api import router as stt_router
from app.stt.api import manager
from app.stt.streaming.asr import FasterWhisperASR

app = FastAPI(title="stt-vietsub-service", version="0.1.0")
logging.basicConfig(level=logging.INFO)
app.include_router(stt_router)

@app.on_event("startup")
async def preload_model() -> None:
    model_size = os.getenv("DEFAULT_MODEL_SIZE", "small")
    device = os.getenv("DEVICE", "cpu")
    compute_type = os.getenv("COMPUTE_TYPE", "int8")
    window_seconds = float(os.getenv("STT_WARMUP_WINDOW_SECONDS", "0.45") or "0.45")
    warmup_windows = int(os.getenv("STT_WARMUP_WINDOWS", "12") or "12")
    logging.getLogger("stt").info(
        "preload model=%s device=%s compute_type=%s", model_size, device, compute_type
    )
    asr = FasterWhisperASR(model_size, device, compute_type, language="vi")
    asr._load_model()
    if warmup_windows > 0 and window_seconds > 0:
        samples = int(16000 * window_seconds)
        silence = np.zeros(samples, dtype=np.float32)
        for _ in range(warmup_windows):
            try:
                asr.transcribe(silence, vad_filter=True, beam_size=1)
            except Exception:
                logging.getLogger("stt").exception("warmup failed")
                break
        logging.getLogger("stt").info(
            "warmup done windows=%s window_seconds=%.2f", warmup_windows, window_seconds
        )


@app.on_event("shutdown")
async def shutdown() -> None:
    await manager.shutdown()

@app.get("/health")
def health():
    return {"ok": True}
