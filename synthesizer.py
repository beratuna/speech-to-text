from __future__ import annotations

import io
import platform
import tempfile
import wave
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

TTS_LANGUAGES: dict[str, str] = {
    "Turkish": "tr",
    "English": "en",
}

MLX_TTS_MODEL = "mlx-community/chatterbox-6bit"
SOFT_TEXT_LIMIT = 600
_TTS_MODEL_LOADED = False


@dataclass(frozen=True)
class SynthesisResult:
    audio_bytes: bytes
    mime_type: str
    sample_rate: int | None
    backend: str


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _has_mlx_tts_support() -> bool:
    if not _is_apple_silicon():
        return False
    try:
        from mlx_audio.tts.utils import load_model as _  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def get_tts_backend_name() -> str:
    return "mlx-audio" if _has_mlx_tts_support() else "chatterbox-tts"


@lru_cache(maxsize=1)
def _load_mlx_tts_model() -> Any:
    from mlx_audio.tts.utils import load_model

    return load_model(MLX_TTS_MODEL)


@lru_cache(maxsize=1)
def _load_chatterbox_model() -> Any:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    return ChatterboxMultilingualTTS.from_pretrained(device="cpu")


def is_tts_model_loaded() -> bool:
    return _TTS_MODEL_LOADED


def load_tts_model() -> None:
    global _TTS_MODEL_LOADED
    if _has_mlx_tts_support():
        _load_mlx_tts_model()
    else:
        _load_chatterbox_model()
    _TTS_MODEL_LOADED = True


def reset_tts_model_state() -> None:
    global _TTS_MODEL_LOADED
    _load_mlx_tts_model.cache_clear()
    _load_chatterbox_model.cache_clear()
    _TTS_MODEL_LOADED = False


def _resolve_generated_wav_path(prefix_path: Path, generated: Any) -> Path:
    candidates: list[Path] = []
    if isinstance(generated, Path | str):
        candidates.append(Path(generated))
    elif isinstance(generated, list | tuple):
        candidates.extend(Path(item) for item in generated if isinstance(item, Path | str))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    direct_path = prefix_path.with_suffix(".wav")
    if direct_path.exists():
        return direct_path

    prefixed_matches = sorted(
        prefix_path.parent.glob(f"{prefix_path.name}*.wav"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if prefixed_matches:
        return prefixed_matches[0]

    all_wavs = sorted(
        prefix_path.parent.glob("*.wav"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if all_wavs:
        return all_wavs[0]

    raise RuntimeError("TTS generation did not produce a WAV file.")


def _generate_mlx_wav_bytes(text: str, language: str, model: Any) -> bytes:
    from mlx_audio.tts.generate import generate_audio

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        prefix_path = output_dir / "tts_output"
        generated = generate_audio(
            model=model,
            text=text,
            lang_code=language,
            file_prefix=str(prefix_path),
        )
        wav_path = _resolve_generated_wav_path(prefix_path=prefix_path, generated=generated)
        return wav_path.read_bytes()


def _wav_bytes_from_tensor(wav_tensor: Any, sample_rate: int) -> bytes:
    import numpy as np

    raw = wav_tensor.detach().cpu().numpy() if hasattr(wav_tensor, "detach") else wav_tensor
    samples = np.asarray(raw)
    if samples.ndim == 2:
        samples = samples[0] if samples.shape[0] <= samples.shape[1] else samples[:, 0]
    elif samples.ndim > 2:
        samples = samples.reshape(-1)

    pcm16 = (np.clip(samples.astype(np.float32), -1.0, 1.0) * 32767.0).astype(np.int16)

    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm16.tobytes())
        return buffer.getvalue()


def _synthesize_mlx(text: str, language: str) -> SynthesisResult:
    model = _load_mlx_tts_model()
    audio_bytes = _generate_mlx_wav_bytes(text=text, language=language, model=model)
    return SynthesisResult(
        audio_bytes=audio_bytes,
        mime_type="audio/wav",
        sample_rate=None,
        backend="mlx-audio",
    )


def _synthesize_chatterbox(text: str, language: str) -> SynthesisResult:
    model = _load_chatterbox_model()
    wav_tensor = model.generate(text, language_id=language)
    sample_rate = int(getattr(model, "sr", 24_000))
    audio_bytes = _wav_bytes_from_tensor(wav_tensor=wav_tensor, sample_rate=sample_rate)
    return SynthesisResult(
        audio_bytes=audio_bytes,
        mime_type="audio/wav",
        sample_rate=sample_rate,
        backend="chatterbox-tts",
    )


def synthesize_with_metadata(text: str, language: str) -> SynthesisResult:
    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("Text cannot be empty.")
    if language not in set(TTS_LANGUAGES.values()):
        raise ValueError(f"Unsupported language code: {language}")

    # TODO: Add cloud fallback backend for constrained deployment environments.
    # TODO: Add chunk-and-concat generation for very long text.
    load_tts_model()
    if _has_mlx_tts_support():
        return _synthesize_mlx(cleaned_text, language=language)
    return _synthesize_chatterbox(cleaned_text, language=language)
