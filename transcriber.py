from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

LANGUAGES: dict[str, str | None] = {
    "Turkish": "tr",
    "English": "en",
    "Auto-detect": None,
}

MODELS: dict[str, str] = {
    "Large v3 Turbo (Default)": "mlx-community/whisper-large-v3-turbo",
    "Medium": "mlx-community/whisper-medium",
    "Small": "mlx-community/whisper-small-mlx",
}

MODEL_CANDIDATES: dict[str, tuple[str, ...]] = {
    "mlx-community/whisper-large-v3-turbo": (
        "mlx-community/whisper-large-v3-turbo",
        "mlx-community/whisper-large-v3",
        "mlx-community/whisper-medium",
    ),
    "mlx-community/whisper-medium": (
        "mlx-community/whisper-medium",
        "mlx-community/whisper-medium-mlx",
        "mlx-community/whisper-base",
    ),
    "mlx-community/whisper-small-mlx": (
        "mlx-community/whisper-small-mlx",
        "mlx-community/whisper-small",
        "mlx-community/whisper-base",
        "mlx-community/whisper-tiny",
    ),
    "mlx-community/whisper-small": (
        "mlx-community/whisper-small",
        "mlx-community/whisper-small-mlx",
        "mlx-community/whisper-base",
        "mlx-community/whisper-tiny",
    ),
}


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str | None = None
    duration_seconds: float | None = None


def _get_mlx_whisper() -> Any:
    try:
        import mlx_whisper  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to import mlx_whisper. Verify MLX setup on this machine."
        ) from exc
    return mlx_whisper


@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> Any:
    mlx_whisper = _get_mlx_whisper()
    load_model_fn = getattr(mlx_whisper, "load_model", None)
    if callable(load_model_fn):
        return load_model_fn(model_path)
    return model_path


def _resolve_model_path(model_path: str) -> str:
    candidates = MODEL_CANDIDATES.get(model_path, (model_path,))
    last_error: Exception | None = None

    for candidate in candidates:
        try:
            load_model(candidate)
            return candidate
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    raise RuntimeError(
        f"Unable to load any model variant for `{model_path}`. Last error: {last_error}"
    ) from last_error


def _audio_duration_seconds(audio_path: Path) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None

    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(audio_path),
    ]
    try:
        output = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None

    try:
        duration = float(output)
    except ValueError:
        return None
    return duration if duration > 0 else None


def _payload_duration_seconds(payload: dict[str, Any]) -> float | None:
    duration = payload.get("duration")
    if isinstance(duration, int | float):
        return float(duration)

    segments = payload.get("segments")
    if not isinstance(segments, list):
        return None

    segment_ends = [
        float(segment["end"])
        for segment in segments
        if isinstance(segment, dict) and isinstance(segment.get("end"), int | float)
    ]
    return max(segment_ends, default=None)


def transcribe_with_metadata(
    audio_path: Path,
    language: str | None,
    model_path: str,
) -> TranscriptionResult:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    resolved_model = _resolve_model_path(model_path=model_path)

    mlx_whisper = _get_mlx_whisper()
    kwargs: dict[str, Any] = {"path_or_hf_repo": resolved_model}
    if language is not None:
        kwargs["language"] = language

    try:
        payload = mlx_whisper.transcribe(str(audio_path), verbose=None, **kwargs)
    except TypeError:
        payload = mlx_whisper.transcribe(str(audio_path), **kwargs)

    if not isinstance(payload, dict):
        return TranscriptionResult(text=str(payload).strip())

    text = str(payload.get("text", "")).strip()
    detected_language = payload.get("language")
    if not isinstance(detected_language, str):
        detected_language = None

    return TranscriptionResult(
        text=text,
        language=detected_language,
        duration_seconds=_audio_duration_seconds(audio_path) or _payload_duration_seconds(payload),
    )


def transcribe(audio_path: Path, language: str | None, model_path: str) -> str:
    return transcribe_with_metadata(
        audio_path=audio_path,
        language=language,
        model_path=model_path,
    ).text
