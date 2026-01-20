# stt-vietsub-service

Dịch vụ STT tiếng Việt realtime-ish sử dụng **faster-whisper**.  
Service nhận audio (RTP/Opus từ **Janus** hoặc **URL**) → decode về PCM → nhận dạng giọng nói → trả text streaming (partial/commit/final).

---

## 1) Chạy nhanh (Local)

### Yêu cầu
- Python **3.11**
- `ffmpeg` đã cài và có trong **PATH**

### Cài đặt & chạy
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e .
python run.py
```

Service chạy tại:
- `http://localhost:9000`


## 2) Biến môi trường (Environment Variables)

Common:
- `HOST`, `PORT`: địa chỉ bind (mặc định `0.0.0.0:9000`)
- `DEVICE`: `cpu` hoặc `cuda`
- `COMPUTE_TYPE`: `int8`, `float16`, ...
- `DEFAULT_MODEL_SIZE`: `small`, `medium`, `large-v2`, ...
- `MAX_SESSIONS`: số session chạy đồng thời

RTP:
- `RTP_HOST`: trả về trong response cho input `janus_rtp_opus`
- `RTP_PORT_MIN`, `RTP_PORT_MAX`: dải port auto allocate

FFmpeg:
- `FFMPEG_LOGLEVEL`: `error`, `warning`, ...
- `FFMPEG_RTBUF_SIZE`, `FFMPEG_BUFFER_SIZE`: tham số buffer của ffmpeg

Tuning Streaming/STT:
- `STT_WINDOW_SECONDS`, `STT_OVERLAP_SECONDS`, `STT_EMIT_INTERVAL_MS`
- `STT_MIN_WINDOW_SECONDS`, `STT_SILENCE_SECONDS`
- `STT_RMS_THRESHOLD`, `STT_MIN_SPEECH_SECONDS`, `STT_MIN_SPEECH_RATIO`
- `STT_BUFFER_SECONDS`, `STT_LAG_DROP_SECONDS`, `STT_LAG_KEEP_SECONDS`
- `STT_PROMPT_CHARS`, `STT_STABLE_MIN_CHARS`
- `STT_BEAM_SIZE`, `STT_CPU_THREADS`, `STT_NUM_WORKERS`
- `STT_NO_SPEECH_THRESHOLD`, `STT_LOGPROB_THRESHOLD`, `STT_COMPRESSION_RATIO_THRESHOLD`

Debug:
- `STT_DUMP_FULL=1` để dump toàn bộ audio ra file `stt_dump_<session_id>.wav`

---

## 3) API

Base: `http://localhost:9000`

### Start session
`POST /v1/sessions`

Các loại input:
- `janus_rtp_opus`: nhận RTP/Opus được forward từ Janus
- `ffmpeg_url`: kéo audio từ URL http/https

### Stop session
`DELETE /v1/sessions/{session_id}`

### List sessions
`GET /v1/sessions`

### Lấy transcript
`GET /v1/sessions/{session_id}/transcript`

---

## 4) Luồng kết nối Janus → STT SUB → Backend (Quan trọng)

### Mục tiêu
**Janus truyền trực tiếp RTP/Opus tới STT SUB** theo `HOST:PORT` chỉ định.  
STT nhận audio từ Janus, xử lý ra text, và trả về backend qua callback.

### Flow realtime chuẩn
- Backend LIVE nhận event `API/RTP/START` từ JANODE  
- Backend LIVE gọi `startRtpForward(...)` để **Janus forward RTP/Opus** tới `STT_HOST:STT_PORT` (SUB)  
- Backend (hoặc SUB) gọi STT `POST /v1/sessions` với input `janus_rtp_opus` để STT bắt đầu listen RTP  
- STT nhận RTP/Opus → decode → transcribe → emit text realtime (partial/commit/final)  
- STT POST caption events về Backend qua `callback.url` (nếu cấu hình)

Sơ đồ:
```
JANODE -> Backend LIVE -> Janus(startRtpForward) -> RTP/Opus -> STT SUB -> Callback(text) -> Backend
```

---

## 5) Logic xử lý Audio → Text (đọc là hiểu)

### A) Pipeline decode audio (RTP/URL -> PCM)
Xử lý nguồn audio nằm ở: `app/stt/ffmpeg_reader.py`

- Với `janus_rtp_opus`:
  - Service tự build SDP để nhận RTP/Opus (payload type mặc định 111)
- Với `ffmpeg_url`:
  - Service đưa URL trực tiếp vào ffmpeg (chỉ http/https)
- ffmpeg output raw PCM chuẩn cho ASR:
  - mono (`-ac 1`)
  - 16kHz (`-ar 16000`)
  - signed 16-bit little-endian (`-f s16le`)
  - `aresample=async=1:first_pts=0` để đồng bộ timestamp
- Output được đọc theo frame **20ms** (640 bytes ở 16kHz mono s16le)

### B) Pipeline text (PCM -> caption streaming)
Core flow nằm ở: `app/stt/worker.py`

1) **Buffering**
   - PCM frames được append vào `AudioBuffer` (giữ history tối đa, mặc định ~40s)

2) **Windowing (cửa sổ trượt)**
   - Tạo sliding window theo:
     - `window_seconds` (độ dài chunk)
     - `overlap_seconds` (độ chồng lấn)
     - `emit_interval_ms` (nhịp loop để “nhả chữ”)

3) **Gate (lọc im lặng / nhiễu)**
   - `SpeechGate` loại các window có RMS thấp hoặc speech ratio quá ít

4) **ASR**
   - Faster-Whisper transcribe window (mặc định `language=vi`)
   - Prompting dùng tail text đã commit để ổn định (`STT_PROMPT_CHARS`)
   - Thresholds: no-speech, log-prob, compression ratio

5) **Ổn định & chống lặp**
   - Stabilize: giữ prefix ổn định khi đủ dài
   - Dedupe: tránh lặp partial/commit
   - Agreement: yêu cầu stable hit đủ số lần trước khi commit

6) **Emit events**
   - `partial`: text realtime (có thể thay đổi)
   - `commit`: text đã chốt (append vào transcript)
   - `final`: snapshot cuối một turn
   - `end`: kết thúc session

Lag control:
- Nếu ingest chạy ahead quá `STT_LAG_DROP_SECONDS`, cursor sẽ fast-forward và chỉ giữ lại `STT_LAG_KEEP_SECONDS` phía sau.

---

## 6) Callback trả text về Backend

Nếu có `callback.url` hoặc env `BACKEND_URL`, STT sẽ POST caption events về callback:
- Header: `x-ai-key` khi `AI_KEY` được set
- Nếu backpressure cao, queue sẽ drop **partials trước** (ưu tiên commit/final)

---

## 7) Troubleshooting

- Không tìm thấy ffmpeg: cài ffmpeg và đảm bảo có trong PATH
- Không ra chữ: kiểm tra `STT_RMS_THRESHOLD` và tăng `window_seconds`
- Lag/chậm: dùng model nhỏ hơn, giảm `STT_WINDOW_SECONDS`, tăng `STT_OVERLAP_SECONDS`
- Quá nhiều session: tăng `MAX_SESSIONS` và tài nguyên CPU/GPU

---

## 8) Cấu trúc project

- `app/main.py`: FastAPI app + warmup model
- `app/stt/api.py`: REST endpoints
- `app/stt/ffmpeg_reader.py`: ffmpeg decode + PCM stream
- `app/stt/worker.py`: streaming ASR pipeline
- `app/stt/streaming/*`: gate, buffer, dedupe, agreement, ASR wrapper

## 9) Các bước chạy
 + Tạo room bắt đầu live (dùng audio màn hoặc audio từ mic)
 + Quay lại log service STT để check quá trình xử lí và text kết quả
