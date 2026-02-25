from __future__ import annotations

import gc
import io
import os
import platform
import tempfile
import warnings
import wave
from contextlib import suppress
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

TTS_LANGUAGES: dict[str, str] = {
    "Turkish": "tr",
    "English": "en",
}

MLX_TTS_MODEL = "mlx-community/chatterbox-6bit"
XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
COQUI_TOS_AGREED_ENV = "COQUI_TOS_AGREED"
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD_ENV = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
SOFT_TEXT_LIMIT = 600
REFERENCE_TARGET_SAMPLE_RATE = 22_050
REFERENCE_MIN_DURATION_SECONDS = 3.0
REFERENCE_MAX_DURATION_SECONDS = 30.0
REFERENCE_SILENCE_TOP_DB = 20
_TTS_MODEL_LOADED = False
_VOICE_CLONE_MODEL_LOADED = False


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


def _has_chatterbox_support() -> bool:
    try:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS as _  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def _has_xtts_support() -> bool:
    try:
        from TTS.api import TTS as _  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def _xtts_device() -> str:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return "cpu"
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    return "mps" if has_mps else "cpu"


def get_tts_backend_name() -> str:
    if _has_mlx_tts_support():
        return "mlx-audio"
    if _has_chatterbox_support():
        return "chatterbox-tts"
    return "gtts"


def is_voice_clone_available() -> bool:
    return _has_xtts_support()


def get_voice_clone_backend_name() -> str:
    if not _has_xtts_support():
        return "unavailable"
    return f"xtts-v2 ({_xtts_device()})"


@lru_cache(maxsize=1)
def _load_mlx_tts_model() -> Any:
    from mlx_audio.tts.utils import load_model

    return load_model(MLX_TTS_MODEL)


@lru_cache(maxsize=1)
def _load_chatterbox_model() -> Any:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    return ChatterboxMultilingualTTS.from_pretrained(device="cpu")


@lru_cache(maxsize=1)
def _load_xtts_model() -> Any:
    from TTS.api import TTS

    return TTS(XTTS_MODEL).to(_xtts_device())


def is_tts_model_loaded() -> bool:
    return _TTS_MODEL_LOADED


def load_tts_model() -> None:
    global _TTS_MODEL_LOADED
    if _has_mlx_tts_support():
        _load_mlx_tts_model()
    elif _has_chatterbox_support():
        _load_chatterbox_model()
    else:
        # gTTS does not require local model loading.
        _TTS_MODEL_LOADED = True
        return
    _TTS_MODEL_LOADED = True


def is_voice_clone_model_loaded() -> bool:
    return _VOICE_CLONE_MODEL_LOADED


def load_voice_clone_model() -> None:
    global _VOICE_CLONE_MODEL_LOADED
    # XTTS v2 prompts interactively for CPML acceptance unless this env is set.
    os.environ.setdefault(COQUI_TOS_AGREED_ENV, "1")
    # Coqui TTS 0.22 internally calls torch.load() without weights_only=False.
    # Torch 2.6 defaults to weights_only=True and breaks XTTS checkpoints.
    os.environ.setdefault(TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD_ENV, "1")
    _load_xtts_model()
    _VOICE_CLONE_MODEL_LOADED = True


def _free_torch_memory() -> None:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    try:
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if has_mps:
            torch.mps.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _move_model_to_cpu(model: Any) -> None:
    if model is None:
        return
    to_fn = getattr(model, "to", None)
    if callable(to_fn):
        with suppress(Exception):
            to_fn("cpu")

    synthesizer = getattr(model, "synthesizer", None)
    tts_model = getattr(synthesizer, "tts_model", None)
    tts_to_fn = getattr(tts_model, "to", None)
    if callable(tts_to_fn):
        with suppress(Exception):
            tts_to_fn("cpu")


def release_standard_tts_model() -> None:
    global _TTS_MODEL_LOADED
    _load_mlx_tts_model.cache_clear()
    _load_chatterbox_model.cache_clear()
    _TTS_MODEL_LOADED = False
    gc.collect()
    _free_torch_memory()


def release_voice_clone_model() -> None:
    global _VOICE_CLONE_MODEL_LOADED
    if _VOICE_CLONE_MODEL_LOADED:
        with suppress(Exception):
            _move_model_to_cpu(_load_xtts_model())
    _load_xtts_model.cache_clear()
    _VOICE_CLONE_MODEL_LOADED = False
    gc.collect()
    _free_torch_memory()


def reset_tts_model_state() -> None:
    global _TTS_MODEL_LOADED, _VOICE_CLONE_MODEL_LOADED
    release_standard_tts_model()
    release_voice_clone_model()


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


def _synthesize_gtts(text: str, language: str) -> SynthesisResult:
    from gtts import gTTS

    with io.BytesIO() as buffer:
        gTTS(text=text, lang=language).write_to_fp(buffer)
        audio_bytes = buffer.getvalue()
    return SynthesisResult(
        audio_bytes=audio_bytes,
        mime_type="audio/mp3",
        sample_rate=None,
        backend="gtts",
    )


def _load_reference_waveform(source_path: Path) -> Any:
    try:
        import librosa
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Failed to import librosa for reference audio preprocessing.") from exc
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="PySoundFile failed. Trying audioread instead.",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="librosa.core.audio.__audioread_load",
            category=FutureWarning,
        )
        waveform, _ = librosa.load(str(source_path), sr=REFERENCE_TARGET_SAMPLE_RATE, mono=True)
    return waveform


def _trim_reference_waveform(waveform: Any) -> Any:
    try:
        import librosa
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Failed to import librosa for reference audio preprocessing.") from exc
    trimmed, _ = librosa.effects.trim(waveform, top_db=REFERENCE_SILENCE_TOP_DB)
    return trimmed


def _write_reference_wav(samples: Any) -> Path:
    try:
        import soundfile as sf
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Failed to import soundfile for reference audio preprocessing.") from exc
    output_path = _create_temp_path(suffix=".wav")
    sf.write(str(output_path), samples, REFERENCE_TARGET_SAMPLE_RATE)
    return output_path


def preprocess_reference_audio(audio_bytes: bytes, suffix: str = ".wav") -> tuple[Path, list[str]]:
    if not audio_bytes:
        raise ValueError("Reference audio file is empty.")

    normalized_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=normalized_suffix) as source_file:
        source_file.write(audio_bytes)
        source_path = Path(source_file.name)

    try:
        waveform = _load_reference_waveform(source_path)
    finally:
        source_path.unlink(missing_ok=True)

    trimmed = _trim_reference_waveform(waveform)
    if len(trimmed) == 0:
        raise ValueError("Reference audio is empty after silence trimming.")

    warnings: list[str] = []
    duration_seconds = len(trimmed) / REFERENCE_TARGET_SAMPLE_RATE
    if duration_seconds < REFERENCE_MIN_DURATION_SECONDS:
        warnings.append(
            f"Reference audio is only {duration_seconds:.1f}s; at least 3s is recommended."
        )
    if duration_seconds > REFERENCE_MAX_DURATION_SECONDS:
        warnings.append(
            f"Reference audio is {duration_seconds:.1f}s; trimming to first 30s for performance."
        )
        max_samples = int(REFERENCE_MAX_DURATION_SECONDS * REFERENCE_TARGET_SAMPLE_RATE)
        trimmed = trimmed[:max_samples]

    return _write_reference_wav(trimmed), warnings


def _validate_synthesis_inputs(text: str, language: str) -> str:
    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("Text cannot be empty.")
    if language not in set(TTS_LANGUAGES.values()):
        raise ValueError(f"Unsupported language code: {language}")
    return cleaned_text


def _create_temp_path(suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        return Path(temp_file.name)


def _synthesize_xtts(text: str, language: str, reference_wav_path: Path) -> SynthesisResult:
    model = _load_xtts_model()
    output_path = _create_temp_path(suffix=".wav")

    try:
        model.tts_to_file(
            text=text,
            speaker_wav=str(reference_wav_path),
            language=language,
            file_path=str(output_path),
        )
        audio_bytes = output_path.read_bytes()
    finally:
        output_path.unlink(missing_ok=True)

    return SynthesisResult(
        audio_bytes=audio_bytes,
        mime_type="audio/wav",
        sample_rate=24_000,
        backend=get_voice_clone_backend_name(),
    )


def synthesize_with_metadata(text: str, language: str) -> SynthesisResult:
    cleaned_text = _validate_synthesis_inputs(text=text, language=language)

    # TODO: Add cloud fallback backend for constrained deployment environments.
    # TODO: Add chunk-and-concat generation for very long text.
    load_tts_model()
    if _has_mlx_tts_support():
        return _synthesize_mlx(cleaned_text, language=language)
    if _has_chatterbox_support():
        return _synthesize_chatterbox(cleaned_text, language=language)
    return _synthesize_gtts(cleaned_text, language=language)


def synthesize_clone_with_metadata(
    text: str,
    language: str,
    reference_wav_path: Path | str,
) -> SynthesisResult:
    cleaned_text = _validate_synthesis_inputs(text=text, language=language)
    reference_path = Path(reference_wav_path)
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference WAV not found: {reference_path}")
    if not is_voice_clone_available():
        raise RuntimeError("Voice clone backend is unavailable. Install the `TTS` package.")

    load_voice_clone_model()
    return _synthesize_xtts(cleaned_text, language=language, reference_wav_path=reference_path)
