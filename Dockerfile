FROM python:3.11-slim

# ffmpeg để decode RTP/Opus -> PCM
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml /app/pyproject.toml
RUN pip install --no-cache-dir uv \
  && uv sync

COPY app /app/app

ENV HOST=0.0.0.0
ENV PORT=9000
ENV DEVICE=cpu
ENV COMPUTE_TYPE=int8
ENV DEFAULT_MODEL_SIZE=small
ENV MAX_SESSIONS=4

CMD ["bash", "-lc", "uvicorn app.main:app --host ${HOST} --port ${PORT}"]
