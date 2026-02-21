from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from transcriber import (
    LANGUAGES,
    TranscriptionResult,
    clear_local_models,
    get_backend_name,
    get_models,
    transcribe_with_metadata,
)

SUPPORTED_TYPES = ["wav", "mp3", "m4a", "ogg", "flac"]
SOFT_WARNING_BYTES = 100 * 1024 * 1024


def _save_temp_audio(content: bytes, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(content)
        return Path(temp_file.name)


def _format_duration(duration_seconds: float | None) -> str:
    if duration_seconds is None:
        return "Unknown"
    rounded = int(round(duration_seconds))
    minutes, seconds = divmod(rounded, 60)
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _render_result(result: TranscriptionResult) -> None:
    st.subheader("Transcription")
    st.text_area("Text", value=result.text, height=260)

    metadata = [f"Audio duration: `{_format_duration(result.duration_seconds)}`"]
    if result.language:
        metadata.insert(0, f"Detected language: `{result.language}`")
    st.caption(" | ".join(metadata))


def _transcribe_audio(audio_path: Path, language: str | None, model_path: str) -> None:
    try:
        with st.spinner("Transcribing..."):
            result = transcribe_with_metadata(
                audio_path=audio_path,
                language=language,
                model_path=model_path,
            )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Transcription failed: {exc}")
        return

    if not result.text:
        st.warning("Transcription completed, but no text was produced.")
        return

    _render_result(result)


def _uploader_tab(language: str | None, model_path: str) -> None:
    uploaded_file = st.file_uploader(
        "Upload audio",
        type=SUPPORTED_TYPES,
        help="Supported formats: wav, mp3, m4a, ogg, flac",
    )
    if uploaded_file is None:
        return

    st.audio(uploaded_file)

    if uploaded_file.size > SOFT_WARNING_BYTES:
        st.warning("Large file detected (>100MB). Transcription may take longer.")

    if not st.button("Transcribe uploaded file", use_container_width=True, key="transcribe_upload"):
        return

    suffix = Path(uploaded_file.name).suffix or ".wav"
    temp_path = _save_temp_audio(uploaded_file.getvalue(), suffix=suffix)
    try:
        _transcribe_audio(temp_path, language=language, model_path=model_path)
    finally:
        temp_path.unlink(missing_ok=True)


def _recorder_tab(language: str | None, model_path: str) -> None:
    recorded_audio = st.audio_input(
        "Record audio",
        help=(
            "Click the mic icon to start recording, and click it again to stop. "
            "Recording does not end automatically on silence."
        ),
    )
    if recorded_audio is None:
        return

    st.audio(recorded_audio)

    if not st.button("Transcribe recording", use_container_width=True, key="transcribe_recording"):
        return

    temp_path = _save_temp_audio(recorded_audio.getvalue(), suffix=".wav")
    try:
        _transcribe_audio(temp_path, language=language, model_path=model_path)
    finally:
        temp_path.unlink(missing_ok=True)


def main() -> None:
    st.set_page_config(page_title="Speech to Text", page_icon="🎙️", layout="wide")
    st.title("Turkish + English Speech to Text")
    st.write("Upload audio or record directly in the browser.")
    st.caption("First run may take longer; model weights download on first use.")

    with st.sidebar:
        st.header("Settings")
        models = get_models()
        selected_model_label = st.selectbox("Model", list(models))
        selected_language_label = st.selectbox("Language", list(LANGUAGES), index=2)
        st.caption(f"Backend: {get_backend_name()}")
        if st.button("Clear local models", use_container_width=True):
            removed, failed = clear_local_models()
            if failed:
                st.warning(f"Removed {removed} model cache directories, {failed} failed.")
            else:
                st.success(f"Removed {removed} model cache directories.")

    model_path = models[selected_model_label]
    language = LANGUAGES[selected_language_label]

    upload_tab, record_tab = st.tabs(["📁 Upload File", "🎤 Record"])
    with upload_tab:
        _uploader_tab(language=language, model_path=model_path)
    with record_tab:
        _recorder_tab(language=language, model_path=model_path)


if __name__ == "__main__":
    main()
