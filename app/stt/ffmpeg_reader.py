import asyncio
import logging
import time
from dataclasses import dataclass
from typing import AsyncIterator, Optional


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
    if input_cfg.type == "janus_rtp_opus":
        sdp = _sdp_for_opus(input_cfg.listen_ip, input_cfg.audio_port, input_cfg.payload_type)
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-protocol_whitelist",
            "file,udp,rtp,pipe",
            "-fflags",
            "+genpts",
            "-i",
            "pipe:0",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "s16le",
            "pipe:1",
        ]
        return FFMpegSource(kind="sdp_pipe", cmd=cmd), sdp

    if input_cfg.type == "ffmpeg_url":
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-fflags",
            "+genpts",
            "-i",
            input_cfg.url,
            "-ac",
            "1",
            "-ar",
            "16000",
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

    try:
        while True:
            if stop_event and stop_event.is_set():
                break
            read_task = asyncio.create_task(proc.stdout.read(4096))
            stop_task = asyncio.create_task(stop_event.wait()) if stop_event else None
            done, _ = await asyncio.wait(
                {read_task, stop_task} if stop_task else {read_task},
                timeout=5.0,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if stop_task and stop_task in done:
                read_task.cancel()
                break
            if read_task not in done:
                read_task.cancel()
                now = time.time()
                if now - last_no_data >= 10.0:
                    logger.warning("ffmpeg no audio data yet")
                    last_no_data = now
                continue
            chunk = read_task.result()
            if not chunk:
                break
            yield chunk
    finally:
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
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
