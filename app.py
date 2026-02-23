from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from synthesizer import (
    SOFT_TEXT_LIMIT,
    TTS_LANGUAGES,
    SynthesisResult,
    get_tts_backend_name,
    is_tts_model_loaded,
    load_tts_model,
    reset_tts_model_state,
    synthesize_with_metadata,
)
from transcriber import (
    LANGUAGES,
    TranscriptionResult,
    clear_local_models,
    get_backend_name,
    get_models,
    is_stt_model_loaded,
    load_stt_model,
    transcribe_with_metadata,
)

SUPPORTED_AUDIO_TYPES = ["wav", "mp3", "m4a", "ogg", "flac"]
SUPPORTED_VIDEO_TYPES = ["mp4", "mov", "m4v", "mkv", "webm", "avi"]
SUPPORTED_MEDIA_TYPES = [*SUPPORTED_AUDIO_TYPES, *SUPPORTED_VIDEO_TYPES]
SOFT_WARNING_BYTES = 100 * 1024 * 1024


def _save_temp_file(content: bytes, suffix: str) -> Path:
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

    metadata = [f"Media duration: `{_format_duration(result.duration_seconds)}`"]
    if result.language:
        metadata.insert(0, f"Detected language: `{result.language}`")
    st.caption(" | ".join(metadata))


def _render_uploaded_media_preview(uploaded_file) -> None:
    extension = Path(uploaded_file.name).suffix.lower().lstrip(".")
    if extension in SUPPORTED_VIDEO_TYPES:
        st.video(uploaded_file)
    else:
        st.audio(uploaded_file)


def _transcribe_audio(audio_path: Path, language: str | None, model_path: str) -> None:
    if not is_stt_model_loaded(model_path):
        try:
            with st.spinner("Loading model..."):
                load_stt_model(model_path)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Model loading failed: {exc}")
            return

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

    st.session_state["latest_transcript"] = result.text
    st.session_state["tts_text"] = result.text
    _render_result(result)


def _uploader_tab(language: str | None, model_path: str) -> None:
    uploaded_file = st.file_uploader(
        "Upload audio or video",
        type=SUPPORTED_MEDIA_TYPES,
        help="Supported audio: wav, mp3, m4a, ogg, flac | video: mp4, mov, m4v, mkv, webm, avi",
    )
    if uploaded_file is None:
        return

    _render_uploaded_media_preview(uploaded_file)

    if uploaded_file.size > SOFT_WARNING_BYTES:
        st.warning("Large file detected (>100MB). Transcription may take longer.")

    if not st.button("Transcribe uploaded file", use_container_width=True, key="transcribe_upload"):
        return

    suffix = Path(uploaded_file.name).suffix or ".tmp"
    temp_path = _save_temp_file(uploaded_file.getvalue(), suffix=suffix)
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

    temp_path = _save_temp_file(recorded_audio.getvalue(), suffix=".wav")
    try:
        _transcribe_audio(temp_path, language=language, model_path=model_path)
    finally:
        temp_path.unlink(missing_ok=True)


def _render_tts_result(result: SynthesisResult) -> None:
    st.audio(result.audio_bytes, format=result.mime_type)
    extension = "mp3" if result.mime_type == "audio/mp3" else "wav"
    st.download_button(
        "Download audio",
        data=result.audio_bytes,
        file_name=f"speech.{extension}",
        mime=result.mime_type,
        use_container_width=True,
    )
    st.caption(f"Backend: {result.backend}")


def _tts_tab() -> None:
    if "tts_text" not in st.session_state:
        st.session_state["tts_text"] = st.session_state.get("latest_transcript", "")

    selected_language_label = st.selectbox(
        "Language", list(TTS_LANGUAGES), key="tts_language_label"
    )
    text_to_speak = st.text_area("Text to speak", key="tts_text", height=220)
    if len(text_to_speak.strip()) > SOFT_TEXT_LIMIT:
        st.warning(
            f"Long text detected (>{SOFT_TEXT_LIMIT} chars). This can be slower; chunking is planned."
        )

    if not st.button("Generate speech", use_container_width=True, key="generate_speech"):
        return

    language_code = TTS_LANGUAGES[selected_language_label]
    if not is_tts_model_loaded():
        try:
            with st.spinner("Loading model..."):
                load_tts_model()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Model loading failed: {exc}")
            return

    try:
        with st.spinner("Generating speech..."):
            result = synthesize_with_metadata(text=text_to_speak, language=language_code)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Speech generation failed: {exc}")
        return

    _render_tts_result(result)


def main() -> None:
    st.set_page_config(page_title="Speech to Text", page_icon="🎙️", layout="wide")
    st.title("Turkish + English Speech to Text")
    st.write("Upload audio/video, record speech, or synthesize text.")
    st.caption("First run may take longer; model weights download on first use.")

    with st.sidebar:
        st.header("Settings")
        selected_task = st.selectbox("Task", ["Speech to Text", "Text to Speech"])

        if selected_task == "Speech to Text":
            models = get_models()
            selected_model_label = st.selectbox("Model", list(models))
            selected_language_label = st.selectbox("Language", list(LANGUAGES), index=2)
            st.caption(f"Backend: {get_backend_name()}")
            model_path = models[selected_model_label]
            language = LANGUAGES[selected_language_label]
        else:
            model_path = None
            language = None
            st.caption(f"Backend: {get_tts_backend_name()}")

        if st.button("Clear local models", use_container_width=True):
            removed, failed = clear_local_models()
            reset_tts_model_state()
            if failed:
                st.warning(f"Removed {removed} model cache directories, {failed} failed.")
            else:
                st.success(f"Removed {removed} model cache directories.")

    if selected_task == "Speech to Text":
        upload_tab, record_tab = st.tabs(["📁 Upload File", "🎤 Record"])
        with upload_tab:
            _uploader_tab(language=language, model_path=model_path)
        with record_tab:
            _recorder_tab(language=language, model_path=model_path)
    else:
        _tts_tab()


if __name__ == "__main__":
    main()
