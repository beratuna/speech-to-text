from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

LANGUAGES: dict[str, str | None] = {
    "Turkish": "tr",
    "English": "en",
    "Auto-detect": None,
}

MODELS_MLX: dict[str, str] = {
    "Large v3 Turbo (Default)": "mlx-community/whisper-large-v3-turbo",
    "Medium": "mlx-community/whisper-medium",
    "Small": "mlx-community/whisper-small-mlx",
}

MODELS_FASTER: dict[str, str] = {
    "Small": "small",
    "Medium": "medium",
}

MODELS_FASTER_REPOS: tuple[str, ...] = (
    "Systran/faster-whisper-large-v3-turbo",
    "Systran/faster-whisper-medium",
    "Systran/faster-whisper-small",
)

_LOADED_STT_MODELS: set[str] = set()


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str | None = None
    duration_seconds: float | None = None


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def get_backend_name() -> str:
    return "mlx-whisper" if _is_apple_silicon() else "faster-whisper"


def get_models() -> dict[str, str]:
    return MODELS_MLX if _is_apple_silicon() else MODELS_FASTER


def clear_local_models() -> tuple[int, int]:
    cache_root = Path(
        os.getenv(
            "HF_HUB_CACHE",
            str(Path.home() / ".cache" / "huggingface" / "hub"),
        )
    )
    removed = 0
    failed = 0

    repo_ids = set(MODELS_MLX.values()) | set(MODELS_FASTER_REPOS)

    for repo_id in repo_ids:
        cache_dir = cache_root / f"models--{repo_id.replace('/', '--')}"
        if not cache_dir.exists():
            continue
        try:
            shutil.rmtree(cache_dir)
            removed += 1
        except OSError:
            failed += 1

    _load_faster_model.cache_clear()
    _load_mlx_model.cache_clear()
    _LOADED_STT_MODELS.clear()
    return removed, failed


def _get_mlx_whisper() -> Any:
    try:
        import mlx_whisper  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to import mlx_whisper. Verify MLX setup on this machine."
        ) from exc
    return mlx_whisper


@lru_cache(maxsize=6)
def _load_faster_model(model_name: str) -> Any:
    from faster_whisper import WhisperModel

    return WhisperModel(model_name, device="cpu", compute_type="int8")


@lru_cache(maxsize=6)
def _load_mlx_model(model_path: str) -> Any:
    mlx_whisper = _get_mlx_whisper()
    load_model_fn = getattr(mlx_whisper, "load_model", None)
    if callable(load_model_fn):
        return load_model_fn(model_path)
    return model_path


def is_stt_model_loaded(model_path: str) -> bool:
    return model_path in _LOADED_STT_MODELS


def load_stt_model(model_path: str) -> None:
    if _is_apple_silicon():
        _load_mlx_model(model_path)
    else:
        _load_faster_model(model_path)
    _LOADED_STT_MODELS.add(model_path)


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


def _segment_duration_seconds(segments: list[Any]) -> float | None:
    segment_ends = [
        float(segment.end)
        for segment in segments
        if isinstance(getattr(segment, "end", None), int | float)
    ]
    return max(segment_ends, default=None)


def transcribe_with_metadata(
    audio_path: Path,
    language: str | None,
    model_path: str,
) -> TranscriptionResult:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if _is_apple_silicon():
        load_stt_model(model_path)
        mlx_whisper = _get_mlx_whisper()
        kwargs: dict[str, Any] = {"path_or_hf_repo": model_path}
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
        duration = _payload_duration_seconds(payload)
    else:
        model = _load_faster_model(model_path)
        _LOADED_STT_MODELS.add(model_path)
        kwargs: dict[str, Any] = {}
        if language is not None:
            kwargs["language"] = language

        segments, info = model.transcribe(str(audio_path), **kwargs)
        segment_list = list(segments)

        text = "".join(segment.text for segment in segment_list).strip()
        detected_language = getattr(info, "language", None)
        if not isinstance(detected_language, str):
            detected_language = None
        duration = _segment_duration_seconds(segment_list)

    return TranscriptionResult(
        text=text,
        language=detected_language,
        duration_seconds=_audio_duration_seconds(audio_path) or duration,
    )


def transcribe(audio_path: Path, language: str | None, model_path: str) -> str:
    return transcribe_with_metadata(
        audio_path=audio_path,
        language=language,
        model_path=model_path,
    ).text
