"""PCM playback regressions using the real HA entity and FFmpeg decoder."""
import asyncio
import math
import shutil
import struct
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from homeassistant.components.tts import TTSAudioRequest
from homeassistant.core import HomeAssistant

from custom_components.openai_tts.exceptions import OpenAINetworkError
from custom_components.openai_tts.tts import OpenAITTSEntity

RATE = 24000
PCM = b"".join(
    struct.pack("<h", round(8000 * math.sin(2 * math.pi * 440 * i / RATE)))
    for i in range(RATE * 2)
)


class SpeechEngine:
    """An upstream raw PCM response with a controllable EOF."""

    _voice = "alloy"
    _model = "tts-1"
    _speed = 1.0

    def __init__(self):
        self.release = asyncio.Event()
        self.release.set()
        self.closed = asyncio.Event()
        self.error = None
        self.requests = []

    async def async_get_tts_stream(self, **kwargs):
        self.requests.append(kwargs)
        try:
            # Deliberately split samples across HTTP chunk boundaries.
            for start in range(0, len(PCM), 4093):
                yield PCM[start:start + 4093]
            await self.release.wait()
            if self.error:
                raise self.error
        finally:
            self.closed.set()

    def get_tts(self, text, **kwargs):
        self.requests.append({"text": text, **kwargs})
        return SimpleNamespace(content=PCM)


@pytest.fixture
async def entity(tmp_path):
    assert shutil.which("ffmpeg"), "These regression tests require FFmpeg"
    hass = HomeAssistant(str(tmp_path))
    config = SimpleNamespace(data={
        "profile_name": "PCM test", "model": "tts-1", "voice": "alloy",
        "audio_format": "pcm", "normalize_audio": False,
        "stream_audio": True, "stream_pipelining": False,
    })
    result = OpenAITTSEntity(hass, config, SpeechEngine())
    result.async_write_ha_state = lambda: None
    result._save_persisted_state = AsyncMock()
    yield result
    await hass.async_stop(force=True)


async def text_stream():
    yield "This is a test announcement."


async def decode(audio, extension):
    """The extension alone must be enough for HA's FFmpeg boundary."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-v", "error", "-f", extension, "-i", "pipe:0",
        "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    pcm, error = await proc.communicate(audio)
    assert proc.returncode == 0, error.decode()
    return pcm


@pytest.mark.asyncio
@pytest.mark.parametrize("pipelining", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
async def test_pcm_is_playable_before_upstream_eof(entity, pipelining, normalize):
    entity._config.data.update(stream_pipelining=pipelining, normalize_audio=normalize)
    engine = entity._engine
    engine.release.clear()
    response = await entity.async_stream_tts_audio(TTSAudioRequest(
        language="en", options={}, message_gen=text_stream(),
    ))
    try:
        # Raw `pcm` is not a container/demuxer understood by HA or browsers.
        assert response.extension == "mp3"
        chunks = []
        while sum(map(len, chunks)) < 1024:
            chunks.append(await asyncio.wait_for(anext(response.data_gen), 3))
        assert not engine.closed.is_set(), "Playback waited for the entire response"
        engine.release.set()
        chunks.extend([chunk async for chunk in response.data_gen])
        decoded = await decode(b"".join(chunks), response.extension)
        # Encoder delay/padding can add frames, but must not change playback speed.
        assert 1.95 < len(decoded) / (RATE * 2) < 2.2
        assert entity._last_duration_ms > 1900
        assert engine.requests[0]["response_format"] == "pcm"
    finally:
        engine.release.set()
        await response.data_gen.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("normalize", [False, True])
async def test_atomic_pcm_is_playable(entity, normalize):
    entity._config.data.update(stream_audio=False, normalize_audio=normalize)
    response = await entity.async_stream_tts_audio(TTSAudioRequest(
        language="en", options={}, message_gen=text_stream(),
    ))
    assert response.extension == "mp3"
    audio = b"".join([chunk async for chunk in response.data_gen])
    decoded = await decode(audio, response.extension)
    assert 1.95 < len(decoded) / (RATE * 2) < 2.2


@pytest.mark.asyncio
async def test_upstream_failure_is_not_cached_as_success(entity):
    entity._engine.error = OpenAINetworkError("connection lost")
    entity._record_stream_success = AsyncMock()
    response = await entity.async_stream_tts_audio(TTSAudioRequest(
        language="en", options={}, message_gen=text_stream(),
    ))
    with pytest.raises(OpenAINetworkError, match="connection lost"):
        async for _ in response.data_gen:
            pass
    entity._record_stream_success.assert_not_awaited()


@pytest.mark.asyncio
async def test_early_close_releases_upstream(entity):
    entity._engine.release.clear()
    response = await entity.async_stream_tts_audio(TTSAudioRequest(
        language="en", options={}, message_gen=text_stream(),
    ))
    await asyncio.wait_for(anext(response.data_gen), 3)
    await response.data_gen.aclose()
    await asyncio.wait_for(entity._engine.closed.wait(), 1)


@pytest.mark.asyncio
async def test_sentence_pipeline_starts_before_final_text(entity):
    entity._config.data["stream_pipelining"] = True
    final_text = asyncio.Event()

    async def messages():
        yield "The first sentence is ready. The next"
        await final_text.wait()
        yield " sentence arrives later."

    response = await entity.async_stream_tts_audio(TTSAudioRequest(
        language="en", options={}, message_gen=messages(),
    ))
    chunks = []
    try:
        while sum(map(len, chunks)) < 1024:
            chunks.append(await asyncio.wait_for(anext(response.data_gen), 3))
        assert len(entity._engine.requests) == 1
        assert entity._engine.requests[0]["text"] == "The first sentence is ready."
        final_text.set()
        chunks.extend([chunk async for chunk in response.data_gen])
        assert len(entity._engine.requests) == 2
        decoded = await decode(b"".join(chunks), response.extension)
        assert 3.95 < len(decoded) / (RATE * 2) < 4.2
    finally:
        final_text.set()
        await response.data_gen.aclose()


@pytest.mark.asyncio
async def test_encoder_failure_is_not_success(entity, monkeypatch):
    monkeypatch.setattr(
        "custom_components.openai_tts.tts.resolve_ffmpeg_paths",
        lambda hass: ("/bin/false", "ffprobe"),
    )
    entity._record_stream_success = AsyncMock()
    response = await entity.async_stream_tts_audio(TTSAudioRequest(
        language="en", options={}, message_gen=text_stream(),
    ))
    with pytest.raises(RuntimeError, match="transcoding failed"):
        async for _ in response.data_gen:
            pass
    entity._record_stream_success.assert_not_awaited()


@pytest.mark.asyncio
async def test_mp3_passthrough_still_has_no_encoder(entity, monkeypatch):
    native = b"ID3" + b"\x00" * 1024

    async def native_stream(**kwargs):
        yield native

    entity._config.data["audio_format"] = "mp3"
    entity._engine.async_get_tts_stream = native_stream
    entity._record_stream_success = AsyncMock()
    spawn = AsyncMock(side_effect=AssertionError("MP3 must remain a passthrough"))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    response = await entity.async_stream_tts_audio(TTSAudioRequest(
        language="en", options={}, message_gen=text_stream(),
    ))
    assert response.extension == "mp3"
    assert b"".join([chunk async for chunk in response.data_gen]) == native
    spawn.assert_not_awaited()


@pytest.mark.asyncio
async def test_chime_and_pcm_share_one_atomic_encoder(entity):
    entity._config.data.update(chime=True, chime_sound="signal1.mp3")
    response = await entity.async_stream_tts_audio(TTSAudioRequest(
        language="en", options={"chime": True}, message_gen=text_stream(),
    ))
    assert response.extension == "mp3"
    decoded = await decode(
        b"".join([chunk async for chunk in response.data_gen]), response.extension,
    )
    assert len(decoded) / (RATE * 2) > 2.2


@pytest.mark.asyncio
async def test_ha_manager_uses_encoded_playback_format(entity, tmp_path, monkeypatch):
    from homeassistant.components.tts import SpeechManager

    def unexpected_conversion(*args, **kwargs):
        raise AssertionError("HA must not convert the MP3 back into raw PCM")

    monkeypatch.setattr(
        "homeassistant.components.tts._async_convert_audio", unexpected_conversion,
    )
    manager = SpeechManager(entity.hass, False, str(tmp_path), 60)
    language, options = manager.process_options(entity, "en", {})
    assert options["audio_format"] == "pcm"
    assert options["preferred_format"] == "mp3"
    stream = manager._async_generate_tts_audio(entity, text_stream(), language, options)
    audio = b"".join([chunk async for chunk in stream])
    decoded = await decode(audio, "mp3")
    assert 1.95 < len(decoded) / (RATE * 2) < 2.2
