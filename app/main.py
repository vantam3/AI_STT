from fastapi import FastAPI
import logging

from app.stt.api import router as stt_router

app = FastAPI(title="stt-vietsub-service", version="0.1.0")
logging.basicConfig(level=logging.INFO)
app.include_router(stt_router)

@app.get("/health")
def health():
    return {"ok": True}
