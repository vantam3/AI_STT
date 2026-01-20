import asyncio
import logging
import time
from dataclasses import dataclass
from typing import AsyncIterator, Optional
from urllib.parse import urlparse

from app.config import settings

@dataclass
class FFMpegSource:
    kind: str
    cmd: list[str]


def _sdp_for_opus(listen_ip: str, port: int, payload_type: int = 111) -> str:
    return "\n".join(
        [
            "v=0",
            f"o=- 0 0 IN IP4 {listen_ip}",
            "s=janus-opus",
            "c=IN IP4 0.0.0.0",
            "t=0 0",
            f"m=audio {port} RTP/AVP {payload_type}",
            f"a=rtpmap:{payload_type} opus/48000/2",
            "a=recvonly",
            "",
        ]
    )


def build_ffmpeg_source(input_cfg) -> tuple[FFMpegSource, str | None]:
    loglevel = settings.FFMPEG_LOGLEVEL
    rtbuf_size = settings.FFMPEG_RTBUF_SIZE
    buffer_size = settings.FFMPEG_BUFFER_SIZE

    async_filter = "aresample=async=1:first_pts=0"

    if input_cfg.type == "janus_rtp_opus":
        sdp = _sdp_for_opus(input_cfg.listen_ip, input_cfg.audio_port, input_cfg.payload_type)
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            loglevel,
            "-protocol_whitelist",
            "file,udp,rtp,pipe",
            "-fflags",
            "+genpts",
        ]
        if rtbuf_size:
            cmd += ["-rtbufsize", rtbuf_size]
        if buffer_size:
            cmd += ["-buffer_size", buffer_size]
        cmd += [
            "-i",
            "pipe:0",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-af",
            async_filter,          
            "-f",
            "s16le",
            "pipe:1",
        ]
        return FFMpegSource(kind="sdp_pipe", cmd=cmd), sdp

    if input_cfg.type == "ffmpeg_url":
        parsed = urlparse(input_cfg.url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("ffmpeg_url must use http or https with a valid host")
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            loglevel,
            "-fflags",
            "+genpts",
        ]
        if rtbuf_size:
            cmd += ["-rtbufsize", rtbuf_size]
        if buffer_size:
            cmd += ["-buffer_size", buffer_size]
        cmd += [
            "-i",
            input_cfg.url,
            "-ac",
            "1",
            "-ar",
            "16000",
            "-af",
            async_filter,          
            "-f",
            "s16le",
            "pipe:1",
        ]
        return FFMpegSource(kind="url", cmd=cmd), None

    raise ValueError(f"unsupported input type: {input_cfg.type}")


async def pcm_stream_from_ffmpeg(
    input_cfg,
    stop_event: Optional[asyncio.Event] = None,
) -> AsyncIterator[bytes]:
    src, sdp = build_ffmpeg_source(input_cfg)

    logger = logging.getLogger("stt")
    logger.info("ffmpeg start kind=%s cmd=%s", src.kind, " ".join(src.cmd))

    proc = await asyncio.create_subprocess_exec(
        *src.cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    if sdp:
        assert proc.stdin is not None
        proc.stdin.write(sdp.encode("utf-8"))
        await proc.stdin.drain()
        proc.stdin.close()
        logger.info("ffmpeg sdp sent")

    assert proc.stdout is not None
    assert proc.stderr is not None

    async def _stderr_logger():
        while True:
            line = await proc.stderr.readline()
            if not line:
                break
            logger.warning("ffmpeg stderr: %s", line.decode(errors="ignore").strip())

    stderr_task = asyncio.create_task(_stderr_logger())

    last_no_data = 0.0
    last_data_ts = time.time()
    read_timeout = settings.STT_FFMPEG_READ_TIMEOUT
    idle_timeout = settings.STT_RTP_IDLE_MS / 1000.0

    frame_bytes = 640  # 20ms @16k mono s16le
    pending = bytearray()

    try:
        while True:
            if stop_event and stop_event.is_set():
                break

            read_task = asyncio.create_task(proc.stdout.read(4096))
            stop_task = asyncio.create_task(stop_event.wait()) if stop_event else None

            done, _ = await asyncio.wait(
                {read_task, stop_task} if stop_task else {read_task},
                timeout=read_timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if stop_task and stop_task not in done:
                stop_task.cancel()

            if stop_task and stop_task in done:
                read_task.cancel()
                break

            if read_task not in done:
                read_task.cancel()
                now = time.time()
                if now - last_no_data >= 10.0:
                    logger.warning("ffmpeg no audio data yet")
                    last_no_data = now
                if idle_timeout > 0 and (now - last_data_ts) >= idle_timeout:
                    logger.info("ffmpeg idle timeout %.3fs reached, stopping", idle_timeout)
                    break
                continue

            chunk = read_task.result()
            if not chunk:
                break
            last_data_ts = time.time()

            pending.extend(chunk)
            while len(pending) >= frame_bytes:
                out = bytes(pending[:frame_bytes])
                del pending[:frame_bytes]
                yield out

    finally:
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=0.5)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()

        if not stderr_task.done():
            stderr_task.cancel()
            try:
                await stderr_task
            except asyncio.CancelledError:
                pass

        logger.info("ffmpeg exit code=%s", proc.returncode)
