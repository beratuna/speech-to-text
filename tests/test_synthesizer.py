from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import synthesizer


def test_get_tts_backend_name_apple(mocker: Any) -> None:
    mocker.patch.object(synthesizer, "_has_mlx_tts_support", return_value=True)

    assert synthesizer.get_tts_backend_name() == "mlx-audio"


def test_get_tts_backend_name_non_apple(mocker: Any) -> None:
    mocker.patch.object(synthesizer, "_has_mlx_tts_support", return_value=False)
    mocker.patch.object(synthesizer, "_has_chatterbox_support", return_value=True)

    assert synthesizer.get_tts_backend_name() == "chatterbox-tts"


def test_get_tts_backend_name_falls_back_to_gtts(mocker: Any) -> None:
    mocker.patch.object(synthesizer, "_has_mlx_tts_support", return_value=False)
    mocker.patch.object(synthesizer, "_has_chatterbox_support", return_value=False)

    assert synthesizer.get_tts_backend_name() == "gtts"


def test_get_voice_clone_backend_name_available(mocker: Any) -> None:
    mocker.patch.object(synthesizer, "_has_xtts_support", return_value=True)
    mocker.patch.object(synthesizer, "_xtts_device", return_value="cpu")

    assert synthesizer.get_voice_clone_backend_name() == "xtts-v2 (cpu)"


def test_get_voice_clone_backend_name_unavailable(mocker: Any) -> None:
    mocker.patch.object(synthesizer, "_has_xtts_support", return_value=False)

    assert synthesizer.get_voice_clone_backend_name() == "unavailable"


def test_load_voice_clone_model_sets_tos_env(mocker: Any) -> None:
    mocker.patch.object(synthesizer, "_load_xtts_model")
    mocker.patch.dict(synthesizer.os.environ, {}, clear=True)
    synthesizer._VOICE_CLONE_MODEL_LOADED = False

    synthesizer.load_voice_clone_model()

    assert synthesizer.os.environ[synthesizer.COQUI_TOS_AGREED_ENV] == "1"
    assert synthesizer.os.environ[synthesizer.TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD_ENV] == "1"
    assert synthesizer.is_voice_clone_model_loaded()


def test_release_voice_clone_model_resets_state(mocker: Any) -> None:
    mocker.patch.object(synthesizer, "_free_torch_memory")
    mocker.patch.object(synthesizer, "_load_xtts_model", return_value=SimpleNamespace(to=lambda *_: None))
    collect = mocker.patch.object(synthesizer.gc, "collect")
    synthesizer._VOICE_CLONE_MODEL_LOADED = True

    synthesizer.release_voice_clone_model()

    assert not synthesizer.is_voice_clone_model_loaded()
    assert collect.called


def test_release_standard_tts_model_resets_state(mocker: Any) -> None:
    mocker.patch.object(synthesizer, "_free_torch_memory")
    collect = mocker.patch.object(synthesizer.gc, "collect")
    synthesizer._TTS_MODEL_LOADED = True

    synthesizer.release_standard_tts_model()

    assert not synthesizer.is_tts_model_loaded()
    assert collect.called


def test_synthesize_non_apple_passes_language_code(mocker: Any) -> None:
    mocker.patch.object(synthesizer, "_has_mlx_tts_support", return_value=False)
    mocker.patch.object(synthesizer, "_has_chatterbox_support", return_value=True)
    fake_model = SimpleNamespace(sr=24_000, generate=mocker.Mock(return_value=object()))
    mocker.patch.object(synthesizer, "_load_chatterbox_model", return_value=fake_model)
    wav_bytes = mocker.patch.object(synthesizer, "_wav_bytes_from_tensor", return_value=b"wav")

    result = synthesizer.synthesize_with_metadata(text="Merhaba dunya", language="tr")

    assert fake_model.generate.call_args.kwargs["language_id"] == "tr"
    assert wav_bytes.call_args.kwargs["sample_rate"] == 24_000
    assert result.backend == "chatterbox-tts"
    assert result.mime_type == "audio/wav"
    assert result.audio_bytes == b"wav"


def test_synthesize_falls_back_to_gtts(mocker: Any) -> None:
    mocker.patch.object(synthesizer, "_has_mlx_tts_support", return_value=False)
    mocker.patch.object(synthesizer, "_has_chatterbox_support", return_value=False)
    gtts = mocker.patch.object(
        synthesizer,
        "_synthesize_gtts",
        return_value=synthesizer.SynthesisResult(
            audio_bytes=b"mp3",
            mime_type="audio/mp3",
            sample_rate=None,
            backend="gtts",
        ),
    )

    result = synthesizer.synthesize_with_metadata(text="Merhaba dunya", language="tr")

    assert gtts.called
    assert result.backend == "gtts"
    assert result.mime_type == "audio/mp3"


def test_synthesize_with_metadata_raises_on_empty_text() -> None:
    with pytest.raises(ValueError, match="Text cannot be empty"):
        synthesizer.synthesize_with_metadata(text="   ", language="en")


def test_preprocess_reference_audio_trims_long_input(mocker: Any, tmp_path: Path) -> None:
    sample_count = int(35 * synthesizer.REFERENCE_TARGET_SAMPLE_RATE)
    waveform = list(range(sample_count))
    mocker.patch.object(synthesizer, "_load_reference_waveform", return_value=waveform)
    mocker.patch.object(synthesizer, "_trim_reference_waveform", return_value=waveform)

    output_path = tmp_path / "reference.wav"
    captured: dict[str, int] = {}

    def _write_stub(samples: Any) -> Path:
        captured["samples"] = len(samples)
        output_path.write_bytes(b"wav")
        return output_path

    mocker.patch.object(synthesizer, "_write_reference_wav", side_effect=_write_stub)

    processed_path, warnings = synthesizer.preprocess_reference_audio(
        audio_bytes=b"fake-audio",
        suffix=".mp3",
    )

    assert processed_path == output_path
    assert "trimming to first 30s" in warnings[0]
    assert captured["samples"] == int(
        synthesizer.REFERENCE_MAX_DURATION_SECONDS * synthesizer.REFERENCE_TARGET_SAMPLE_RATE
    )


def test_synthesize_clone_with_metadata_calls_xtts_path(mocker: Any, tmp_path: Path) -> None:
    reference_path = tmp_path / "speaker.wav"
    reference_path.write_bytes(b"wav")
    expected = synthesizer.SynthesisResult(
        audio_bytes=b"xtts",
        mime_type="audio/wav",
        sample_rate=24_000,
        backend="xtts-v2 (cpu)",
    )

    mocker.patch.object(synthesizer, "is_voice_clone_available", return_value=True)
    load = mocker.patch.object(synthesizer, "load_voice_clone_model")
    clone = mocker.patch.object(synthesizer, "_synthesize_xtts", return_value=expected)

    result = synthesizer.synthesize_clone_with_metadata(
        text="Merhaba dunya",
        language="tr",
        reference_wav_path=reference_path,
    )

    load.assert_called_once()
    assert clone.call_args.kwargs["reference_wav_path"] == reference_path
    assert result == expected


def test_synthesize_clone_with_metadata_raises_when_backend_missing(mocker: Any, tmp_path: Path) -> None:
    reference_path = tmp_path / "speaker.wav"
    reference_path.write_bytes(b"wav")
    mocker.patch.object(synthesizer, "is_voice_clone_available", return_value=False)

    with pytest.raises(RuntimeError, match="Voice clone backend is unavailable"):
        synthesizer.synthesize_clone_with_metadata(
            text="Merhaba dunya",
            language="tr",
            reference_wav_path=reference_path,
        )
