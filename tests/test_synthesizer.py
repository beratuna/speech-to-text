from __future__ import annotations

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
