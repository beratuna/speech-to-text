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


@pytest.fixture
def fake_mlx(mocker: Any) -> Any:
    mocker.patch.object(transcriber, "_is_apple_silicon", return_value=True)
    fake_module = SimpleNamespace(
        transcribe=mocker.Mock(return_value={"text": "ok"}),
    )
    mocker.patch.object(transcriber, "_get_mlx_whisper", return_value=fake_module)
    return fake_module


def test_get_models_apple_silicon(mocker: Any) -> None:
    mocker.patch.object(transcriber, "_is_apple_silicon", return_value=True)

    assert transcriber.get_models() == transcriber.MODELS_MLX


def test_get_models_non_apple(mocker: Any) -> None:
    mocker.patch.object(transcriber, "_is_apple_silicon", return_value=False)

    assert transcriber.get_models() == transcriber.MODELS_FASTER
    assert "Large v3 Turbo (Default)" not in transcriber.get_models()


def test_transcribe_returns_stripped_text(fake_mlx: Any, wav_file: Path) -> None:
    fake_mlx.transcribe.return_value = {"text": "  merhaba world  "}

    result = transcriber.transcribe(wav_file, language="tr", model_path="model-id")

    assert result == "merhaba world"


def test_transcribe_passes_language_code(fake_mlx: Any, wav_file: Path) -> None:
    transcriber.transcribe(wav_file, language="en", model_path="model-id")

    assert fake_mlx.transcribe.call_args.kwargs["language"] == "en"


def test_transcribe_passes_model_path(fake_mlx: Any, wav_file: Path) -> None:
    transcriber.transcribe(wav_file, language=None, model_path="mlx-community/whisper-small-mlx")

    assert (
        fake_mlx.transcribe.call_args.kwargs["path_or_hf_repo"] == "mlx-community/whisper-small-mlx"
    )


def test_transcribe_uses_faster_whisper_non_apple(mocker: Any, wav_file: Path) -> None:
    mocker.patch.object(transcriber, "_is_apple_silicon", return_value=False)
    fake_model = SimpleNamespace(
        transcribe=mocker.Mock(
            return_value=(
                iter(
                    [
                        SimpleNamespace(text=" hello", end=1.0),
                        SimpleNamespace(text=" world ", end=2.0),
                    ]
                ),
                SimpleNamespace(language="en"),
            )
        )
    )
    load_model = mocker.patch.object(transcriber, "_load_faster_model", return_value=fake_model)

    result = transcriber.transcribe_with_metadata(wav_file, language="en", model_path="small")

    assert result.text == "hello world"
    assert result.language == "en"
    assert load_model.call_args.args[0] == "small"
    assert fake_model.transcribe.call_args.kwargs["language"] == "en"


def test_clear_local_models_removes_hf_model_cache_dirs(tmp_path: Path, mocker: Any) -> None:
    mocker.patch.dict("os.environ", {"HF_HUB_CACHE": str(tmp_path)})
    target_repo = transcriber.MODELS_MLX["Small"]
    target_dir = tmp_path / f"models--{target_repo.replace('/', '--')}"
    target_dir.mkdir(parents=True)

    removed, failed = transcriber.clear_local_models()

    assert removed == 1
    assert failed == 0
    assert not target_dir.exists()


@pytest.mark.integration
def test_integration_transcribe_sample_file() -> None:
    sample_audio = Path(__file__).resolve().parents[1] / "vize workflow.m4a"
    if not sample_audio.exists():
        pytest.skip("Sample audio file not found.")

    default_model = transcriber.get_models()["Small"]
    text = transcriber.transcribe(
        audio_path=sample_audio,
        language="tr",
        model_path=default_model,
    )

    assert isinstance(text, str)
    assert text.strip()
