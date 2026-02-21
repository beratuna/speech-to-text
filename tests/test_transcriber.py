from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import transcriber


def _write_minimal_wav(path: Path) -> None:
    # Minimal valid 44-byte WAV header.
    path.write_bytes(
        b"RIFF$\x00\x00\x00WAVEfmt "
        b"\x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00"
        b"\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    )


@pytest.fixture
def wav_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "sample.wav"
    _write_minimal_wav(file_path)
    return file_path


@pytest.fixture(autouse=True)
def clear_model_cache() -> None:
    transcriber.load_model.clear()


@pytest.fixture
def fake_mlx(mocker: Any) -> Any:
    fake_module = SimpleNamespace(
        load_model=mocker.Mock(return_value=object()),
        transcribe=mocker.Mock(return_value={"text": "ok"}),
    )
    mocker.patch.object(transcriber, "_get_mlx_whisper", return_value=fake_module)
    return fake_module


def test_transcribe_returns_stripped_text(fake_mlx: Any, wav_file: Path) -> None:
    fake_mlx.transcribe.return_value = {"text": "  merhaba world  "}

    result = transcriber.transcribe(wav_file, language="tr", model_path="model-id")

    assert result == "merhaba world"


def test_transcribe_passes_language_code(fake_mlx: Any, wav_file: Path) -> None:
    transcriber.transcribe(wav_file, language="en", model_path="model-id")

    assert fake_mlx.transcribe.call_args.kwargs["language"] == "en"


def test_transcribe_passes_model_path(fake_mlx: Any, wav_file: Path) -> None:
    transcriber.transcribe(wav_file, language=None, model_path="mlx-community/whisper-small")

    fake_mlx.load_model.assert_called_once_with("mlx-community/whisper-small")


def test_load_model_caches(fake_mlx: Any) -> None:
    sentinel_model = object()
    fake_mlx.load_model.return_value = sentinel_model

    first = transcriber.load_model("mlx-community/whisper-small")
    second = transcriber.load_model("mlx-community/whisper-small")

    assert first is sentinel_model
    assert second is sentinel_model
    fake_mlx.load_model.assert_called_once_with("mlx-community/whisper-small")


def test_resolve_model_path_falls_back_when_primary_fails(mocker: Any) -> None:
    attempts: list[str] = []

    def _fake_load(model_path: str) -> object:
        attempts.append(model_path)
        if model_path == "mlx-community/whisper-small":
            raise RuntimeError("small repo not available")
        return object()

    mocker.patch.object(transcriber, "load_model", side_effect=_fake_load)

    resolved = transcriber._resolve_model_path(model_path="mlx-community/whisper-small")

    assert resolved == "mlx-community/whisper-small-mlx"
    assert attempts == [
        "mlx-community/whisper-small",
        "mlx-community/whisper-small-mlx",
    ]


@pytest.mark.integration
def test_integration_transcribe_sample_file() -> None:
    sample_audio = Path(__file__).resolve().parents[1] / "vize workflow.m4a"
    if not sample_audio.exists():
        pytest.skip("Sample audio file not found.")

    default_model = transcriber.MODELS["Small"]
    text = transcriber.transcribe(
        audio_path=sample_audio,
        language="tr",
        model_path=default_model,
    )

    assert isinstance(text, str)
    assert text.strip()
