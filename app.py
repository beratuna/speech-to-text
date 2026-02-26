from __future__ import annotations

import os
import warnings

import streamlit as st

from synthesizer import (
    SOFT_TEXT_LIMIT,
    TTS_LANGUAGES,
    SynthesisResult,
    get_tts_backend_name,
    get_voice_clone_backend_name,
    is_tts_model_loaded,
    is_voice_clone_available,
    is_voice_clone_model_loaded,
    load_tts_model,
    load_voice_clone_model,
    preprocess_reference_audio,
    release_standard_tts_model,
    release_voice_clone_model,
    reset_tts_model_state,
    synthesize_clone_with_metadata,
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
    release_stt_models,
    transcribe_with_metadata,
)
from workflow_media import (
    MediaPayload,
    render_media_source,
    save_media_payload,
    to_reference_audio_payload,
)

IS_STREAMLIT_CLOUD = os.environ.get("HOME") == "/home/appuser"
ALLOW_CLOUD_VOICE_CLONE = os.environ.get("ALLOW_CLOUD_VOICE_CLONE") == "1"

WORKFLOW_SPEECH = "Speech -> Text/Speech"
WORKFLOW_TEXT_TO_SPEECH = "Text -> Speech"

TTS_MODE_STANDARD = "Standard"
TTS_MODE_CLONE = "Voice Clone"
OUTPUT_TEXT_ONLY = "Text only"

warnings.filterwarnings(
    "ignore",
    message="Torchaudio's I/O functions now support par-call backend dispatch.*",
    category=UserWarning,
)


def _inject_ui_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ui-text: #1a2436;
            --ui-muted: #5e6f88;
            --ui-card: rgba(255, 255, 255, 0.9);
            --ui-border: #d7dfef;
            --ui-soft: #f4f7ff;
            --ui-accent: #2f6bff;
            --ui-dashed: #c9d3ea;
            --ui-control-bg: #697386;
            --ui-control-bg-hover: #7b8496;
            --ui-control-border: #8e97a8;
            --ui-control-border-strong: #a2abbb;
            --ui-control-fg: #eef3ff;
            --ui-segment-active-bg: rgba(236, 136, 145, 0.24);
            --ui-segment-active-border: rgba(209, 96, 106, 0.52);
            --ui-segment-active-fg: #4a1f24;
            --ui-segment-hover-bg: #7e889a;
            --ui-segment-hover-border: #b0bacb;
        }
        html, body, [data-testid="stAppViewContainer"], .stApp {
            background:
                radial-gradient(1200px 460px at -10% -10%, #dce7ff 0%, transparent 60%),
                radial-gradient(900px 420px at 110% 0%, #ffe7d4 0%, transparent 58%),
                linear-gradient(180deg, #f8faff 0%, #eff3fb 100%);
            color: var(--ui-text);
        }
        [data-testid="stHeader"] {
            background: transparent;
        }
        .block-container {
            max-width: 1060px;
            padding-top: 1.4rem;
            padding-bottom: 2.8rem;
        }
        .stMarkdown, .stCaption, label, p, span {
            color: var(--ui-text);
        }
        .ui-hero {
            background: var(--ui-card);
            border: 1px solid var(--ui-border);
            border-radius: 18px;
            padding: 1.15rem 1.25rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 10px 26px rgba(26, 36, 54, 0.08);
            backdrop-filter: blur(4px);
        }
        .ui-hero h1 {
            margin: 0;
            font-size: 2.0rem;
            letter-spacing: -0.02em;
        }
        .ui-hero p {
            margin: 0.35rem 0 0;
            color: var(--ui-muted);
            font-size: 1rem;
        }
        .section-kicker {
            margin: 0.4rem 0 0.2rem;
            color: var(--ui-muted);
            font-size: 0.92rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            text-transform: uppercase;
        }
        .ui-muted {
            margin: 0.25rem 0 0.45rem;
            color: var(--ui-muted);
        }
        .media-option-card {
            border: 1px dashed var(--ui-dashed);
            border-radius: 14px;
            background: var(--ui-soft);
            padding: 1rem;
            text-align: center;
            min-height: 150px;
            margin-bottom: 0.45rem;
        }
        .media-option-icon {
            font-size: 2rem;
            line-height: 1;
            margin-bottom: 0.45rem;
        }
        .media-option-title {
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .media-option-caption {
            font-size: 0.92rem;
            color: var(--ui-muted);
        }
        .stTextArea textarea,
        .stTextInput input,
        [data-baseweb="select"] > div {
            background: #ffffff !important;
            color: var(--ui-text) !important;
            border-color: var(--ui-border) !important;
            border-radius: 10px !important;
        }
        [data-testid="stFileUploaderDropzone"] {
            background: var(--ui-control-bg) !important;
            border: 1px dashed var(--ui-control-border) !important;
            border-radius: 12px !important;
        }
        [data-testid="stFileUploaderDropzone"],
        [data-testid="stFileUploaderDropzone"] * {
            color: var(--ui-control-fg) !important;
        }
        [data-testid="stFileUploader"] button,
        [data-testid="stFileUploader"] [data-testid^="stBaseButton"] {
            background: var(--ui-control-bg-hover) !important;
            color: var(--ui-control-fg) !important;
            border: 1px solid var(--ui-control-border-strong) !important;
        }
        [data-testid="stAudioInput"] > div,
        [data-testid="stAudioInput"] .e18uw4vz1,
        div.e18uw4vz1 {
            background: var(--ui-control-bg) !important;
            border: 1px dashed var(--ui-control-border) !important;
            border-radius: 12px !important;
        }
        [data-testid="stAudioInput"] *,
        [data-testid="stAudioInput"] .e18uw4vz1 *,
        div.e18uw4vz1 * {
            color: var(--ui-control-fg) !important;
        }
        [data-testid="stAudioInput"] button {
            background: var(--ui-control-bg-hover) !important;
            color: var(--ui-control-fg) !important;
            border: 1px solid var(--ui-control-border-strong) !important;
        }
        [data-testid="stAudioInput"] [aria-live="polite"],
        [data-testid="stAudioInput"] [role="status"],
        [data-testid="stAudioInput"] [role="timer"] {
            display: none !important;
        }
        [data-testid="stAudioInputWaveformTimeCode"],
        [data-testid="stAudioInput"] .e18uw4vz4,
        span.e18uw4vz4 {
            display: none !important;
        }
        [data-baseweb="button-group"],
        div[role="radiogroup"] {
            background: transparent !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0 !important;
        }
        [data-testid="stExpander"] details {
            border: 1px solid #8e97a8 !important;
            border-radius: 12px !important;
            overflow: hidden;
        }
        [data-testid="stExpander"] summary,
        summary.st-emotion-cache-11ofl8m.e12o48ov4 {
            background: var(--ui-control-bg) !important;
            color: var(--ui-control-fg) !important;
        }
        [data-testid="stExpander"] summary:hover,
        summary.st-emotion-cache-11ofl8m.e12o48ov4:hover {
            background: var(--ui-control-bg-hover) !important;
            color: var(--ui-control-fg) !important;
        }
        [data-testid="stExpander"] summary *,
        summary.st-emotion-cache-11ofl8m.e12o48ov4 * {
            color: var(--ui-control-fg) !important;
        }
        [data-testid="stCheckbox"] [role="checkbox"] {
            background: #6f7889 !important;
            border: 1px solid #9aa4b6 !important;
            border-radius: 6px !important;
        }
        [data-testid="stCheckbox"] [role="checkbox"][aria-checked="true"] {
            background: #858fa0 !important;
            border-color: #aeb7c7 !important;
        }
        [data-testid="stCheckbox"] [role="checkbox"] svg {
            fill: #eef3ff !important;
        }
        button[data-testid="stBaseButton-segmented_control"],
        button[data-testid="stBaseButton-segmented_controlActive"],
        button.e1mwqyj912,
        button.e1mwqyj913 {
            background: var(--ui-control-bg) !important;
            color: var(--ui-control-fg) !important;
            border: 1px solid var(--ui-control-border) !important;
            transition: background 140ms ease, color 140ms ease, border-color 140ms ease;
        }
        button[data-testid="stBaseButton-segmented_control"] *,
        button[data-testid="stBaseButton-segmented_controlActive"] *,
        button.e1mwqyj912 *,
        button.e1mwqyj913 * {
            color: var(--ui-control-fg) !important;
        }
        button[data-testid="stBaseButton-segmented_controlActive"],
        button[kind="segmented_controlActive"],
        button[data-testid="stBaseButton-segmented_control"][aria-selected="true"],
        button[data-testid="stBaseButton-segmented_control"][aria-pressed="true"],
        button[data-testid="stBaseButton-segmented_control"][data-selected="true"],
        button[kind="segmented_control"][aria-selected="true"],
        button[kind="segmented_control"][aria-pressed="true"],
        button.e1mwqyj913 {
            background: var(--ui-segment-active-bg) !important;
            border-color: var(--ui-segment-active-border) !important;
            color: var(--ui-segment-active-fg) !important;
            box-shadow: none !important;
        }
        button[data-testid="stBaseButton-segmented_controlActive"] *,
        button[kind="segmented_controlActive"] *,
        button[data-testid="stBaseButton-segmented_control"][aria-selected="true"] *,
        button[data-testid="stBaseButton-segmented_control"][aria-pressed="true"] *,
        button[data-testid="stBaseButton-segmented_control"][data-selected="true"] *,
        button[kind="segmented_control"][aria-selected="true"] *,
        button[kind="segmented_control"][aria-pressed="true"] *,
        button.e1mwqyj913 * {
            color: var(--ui-segment-active-fg) !important;
            font-weight: 600 !important;
        }
        button[data-testid="stBaseButton-segmented_control"]:not([aria-selected="true"]):not([aria-pressed="true"]):not([data-selected="true"]):hover,
        button.e1mwqyj912:hover {
            background: var(--ui-segment-hover-bg) !important;
            border-color: var(--ui-segment-hover-border) !important;
            color: var(--ui-control-fg) !important;
        }
        button[data-testid="stBaseButton-segmented_control"]:not([aria-selected="true"]):not([aria-pressed="true"]):not([data-selected="true"]):hover *,
        button.e1mwqyj912:hover * {
            color: var(--ui-control-fg) !important;
        }
        button[data-testid="stBaseButton-segmented_controlActive"]:hover,
        button[kind="segmented_controlActive"]:hover,
        button[data-testid="stBaseButton-segmented_control"][aria-selected="true"]:hover,
        button[data-testid="stBaseButton-segmented_control"][aria-pressed="true"]:hover,
        button[data-testid="stBaseButton-segmented_control"][data-selected="true"]:hover,
        button[kind="segmented_control"][aria-selected="true"]:hover,
        button[kind="segmented_control"][aria-pressed="true"]:hover,
        button.e1mwqyj913:hover {
            background: var(--ui-segment-active-bg) !important;
            border-color: var(--ui-segment-active-border) !important;
            color: var(--ui-segment-active-fg) !important;
        }
        button[data-testid="stBaseButton-segmented_controlActive"]:hover *,
        button[kind="segmented_controlActive"]:hover *,
        button[data-testid="stBaseButton-segmented_control"][aria-selected="true"]:hover *,
        button[data-testid="stBaseButton-segmented_control"][aria-pressed="true"]:hover *,
        button[data-testid="stBaseButton-segmented_control"][data-selected="true"]:hover *,
        button[kind="segmented_control"][aria-selected="true"]:hover *,
        button[kind="segmented_control"][aria-pressed="true"]:hover *,
        button.e1mwqyj913:hover * {
            color: var(--ui-segment-active-fg) !important;
        }
        .stButton > button {
            border-radius: 11px;
            border-color: #c9d6f3;
            background: #ffffff;
            color: var(--ui-text);
            font-weight: 600;
        }
        .stButton > button:hover {
            border-color: #9db6f8;
            color: #1749c9;
        }
        .media-option-divider {
            text-align: center;
            color: var(--ui-muted);
            font-size: 1.0rem;
            padding-top: 3.8rem;
        }
        .media-guidance-box {
            margin-top: 0.55rem;
            border: 1px solid var(--ui-border);
            background: var(--ui-soft);
            border-radius: 12px;
            padding: 0.75rem 0.85rem;
            color: var(--ui-muted);
            font-size: 0.92rem;
            line-height: 1.4;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_page_header() -> None:
    st.markdown(
        """
        <div class="ui-hero">
          <h1>Voice Cloning Studio</h1>
          <p>Import speech, extract text, and generate standard or cloned voice output in one flow.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _format_duration(duration_seconds: float | None) -> str:
    if duration_seconds is None:
        return "Unknown"
    rounded = int(round(duration_seconds))
    minutes, seconds = divmod(rounded, 60)
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _render_transcription_summary(result: TranscriptionResult) -> None:
    metadata = [f"Media duration: `{_format_duration(result.duration_seconds)}`"]
    if result.language:
        metadata.insert(0, f"Detected language: `{result.language}`")
    st.caption(" | ".join(metadata))


def _render_tts_result(result: SynthesisResult) -> None:
    st.audio(result.audio_bytes, format=result.mime_type)
    extension = "mp3" if result.mime_type == "audio/mp3" else "wav"
    st.download_button(
        "Download speech",
        data=result.audio_bytes,
        file_name=f"speech.{extension}",
        mime=result.mime_type,
        use_container_width=True,
    )
    st.caption(f"Backend: {result.backend}")


def _render_voice_clone_tips() -> None:
    with st.expander("Voice clone tips"):
        st.markdown(
            "- **Duration:** 5-15 seconds is ideal.\n"
            "- **Content:** Clear, natural speech works best.\n"
            "- **Noise:** Quiet recordings clone better.\n"
            "- **Language match:** Matching source/target language usually improves quality."
        )


def _show_error(prefix: str, exc: Exception) -> None:
    details = str(exc).strip() or exc.__class__.__name__
    st.error(f"{prefix}: {details}")
    with st.expander("Technical details"):
        st.exception(exc)


def _release_all_models() -> None:
    release_stt_models()
    release_standard_tts_model()
    release_voice_clone_model()


def _handle_workflow_switch(selected_workflow: str) -> None:
    previous_workflow = st.session_state.get("active_workflow")
    if previous_workflow == selected_workflow:
        return
    st.session_state["active_workflow"] = selected_workflow
    _release_all_models()


def _prepare_single_segment_value(
    key: str,
    options: list[str],
    default: str,
) -> str:
    last_valid_key = f"{key}__last_valid"
    candidate = st.session_state.get(key)
    if isinstance(candidate, (list, tuple)):
        candidate = next((item for item in candidate if item in options), None)

    if candidate not in options:
        previous_valid = st.session_state.get(last_valid_key)
        candidate = previous_valid if previous_valid in options else default

    st.session_state[key] = candidate
    st.session_state[last_valid_key] = candidate
    return candidate


def _resolve_single_segment_value(
    key: str,
    options: list[str],
    default: str,
    value: str | list[str] | tuple[str, ...] | None,
) -> str:
    last_valid_key = f"{key}__last_valid"
    restore_flag_key = f"{key}__restore_pending"

    candidate = value
    if isinstance(candidate, (list, tuple)):
        candidate = next((item for item in candidate if item in options), None)

    previous_valid = st.session_state.get(last_valid_key)
    fallback = previous_valid if previous_valid in options else default

    if candidate in options:
        st.session_state[last_valid_key] = candidate
        st.session_state[restore_flag_key] = False
        return candidate

    if not st.session_state.get(restore_flag_key, False):
        st.session_state[restore_flag_key] = True
        st.rerun()

    st.session_state[restore_flag_key] = False
    return fallback


def _transcribe_source_media(
    source_media: MediaPayload,
    language: str | None,
    model_path: str,
) -> TranscriptionResult:
    source_path = save_media_payload(source_media)
    try:
        if not is_stt_model_loaded(model_path):
            with st.spinner("Loading transcription model..."):
                load_stt_model(model_path)
        with st.spinner("Transcribing speech..."):
            result = transcribe_with_metadata(
                audio_path=source_path,
                language=language,
                model_path=model_path,
            )
    finally:
        source_path.unlink(missing_ok=True)

    if not result.text:
        raise ValueError("Transcription completed, but no text was produced.")

    st.session_state["latest_transcript"] = result.text
    st.session_state["workflow_text"] = result.text
    return result


def _generate_standard_tts(text: str, language_code: str) -> SynthesisResult:
    if is_voice_clone_model_loaded():
        release_voice_clone_model()
    if not is_tts_model_loaded():
        with st.spinner("Loading speech model..."):
            load_tts_model()
    with st.spinner("Generating speech..."):
        return synthesize_with_metadata(text=text, language=language_code)


def _generate_clone_tts(
    text: str,
    language_code: str,
    reference_media: MediaPayload,
    keep_clone_model_loaded: bool,
) -> SynthesisResult:
    if is_tts_model_loaded():
        release_standard_tts_model()
    if not is_voice_clone_available():
        raise RuntimeError("Voice clone backend is unavailable. Install the `TTS` dependency.")

    ref_path = None
    try:
        reference_audio_bytes, reference_suffix = to_reference_audio_payload(reference_media)
        with st.spinner("Preprocessing reference audio..."):
            ref_path, preprocess_warnings = preprocess_reference_audio(
                audio_bytes=reference_audio_bytes,
                suffix=reference_suffix,
            )
        for warning in preprocess_warnings:
            st.warning(warning)

        if not is_voice_clone_model_loaded():
            with st.spinner("Loading clone model..."):
                load_voice_clone_model()

        with st.spinner("Cloning voice and generating speech..."):
            return synthesize_clone_with_metadata(
                text=text,
                language=language_code,
                reference_wav_path=ref_path,
            )
    finally:
        if ref_path is not None:
            ref_path.unlink(missing_ok=True)
        if not keep_clone_model_loaded:
            release_voice_clone_model()


def _generate_tts_from_text(
    text: str,
    tts_mode: str,
    language_code: str,
    reference_media: MediaPayload | None,
    keep_clone_model_loaded: bool,
) -> SynthesisResult:
    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("Text cannot be empty.")
    if len(cleaned_text) > SOFT_TEXT_LIMIT:
        st.warning(
            f"Long text detected (>{SOFT_TEXT_LIMIT} chars). Generation can be slower for long text."
        )

    release_stt_models()

    if tts_mode == TTS_MODE_STANDARD:
        return _generate_standard_tts(cleaned_text, language_code=language_code)
    if reference_media is None:
        raise ValueError("Reference speech is required for Voice Clone mode.")
    return _generate_clone_tts(
        cleaned_text,
        language_code=language_code,
        reference_media=reference_media,
        keep_clone_model_loaded=keep_clone_model_loaded,
    )


def _render_maintenance_section() -> None:
    with st.expander("Maintenance"):
        st.caption(f"STT backend: {get_backend_name()}")
        st.caption(f"Standard TTS backend: {get_tts_backend_name()}")
        st.caption(f"Voice clone backend: {get_voice_clone_backend_name()}")
        if st.button("Clear local model caches", use_container_width=True):
            removed, failed = clear_local_models()
            reset_tts_model_state()
            if failed:
                st.warning(f"Removed {removed} cache directories, {failed} failed.")
            else:
                st.success(f"Removed {removed} cache directories.")


def main() -> None:
    st.set_page_config(page_title="Voice Cloning Studio", page_icon="🎙️", layout="wide")
    _inject_ui_styles()
    _render_page_header()

    workflows = [WORKFLOW_SPEECH, WORKFLOW_TEXT_TO_SPEECH]
    _prepare_single_segment_value("workflow_goal", workflows, WORKFLOW_SPEECH)
    selected_workflow = st.segmented_control(
        "Workflow",
        workflows,
        selection_mode="single",
        key="workflow_goal",
    )
    selected_workflow = _resolve_single_segment_value(
        "workflow_goal",
        workflows,
        WORKFLOW_SPEECH,
        selected_workflow,
    )
    st.caption(
        "Default flow is speech to speech. Switch output mode to Text only if you just want transcription."
    )
    _handle_workflow_switch(selected_workflow)

    is_speech_flow = selected_workflow == WORKFLOW_SPEECH

    _render_maintenance_section()

    source_media: MediaPayload | None = None
    model_path = ""
    transcription_language: str | None = None
    if is_speech_flow:
        st.markdown("<p class='section-kicker'>Import Your Voice</p>", unsafe_allow_html=True)
        source_media = render_media_source(
            title="Source Speech",
            key_prefix="source_media",
            helper_text=(
                "Add a file or record speech directly. Audio and video are both supported."
            ),
        )
        st.markdown("<p class='section-kicker'>Transcription Settings</p>", unsafe_allow_html=True)
        models = get_models()
        model_col, lang_col = st.columns(2)
        with model_col:
            selected_model_label = st.selectbox("Model", list(models), key="workflow_stt_model")
        with lang_col:
            selected_language_label = st.selectbox(
                "Language", list(LANGUAGES), index=2, key="workflow_stt_language"
            )
        model_path = models[selected_model_label]
        transcription_language = LANGUAGES[selected_language_label]

    if "workflow_text" not in st.session_state:
        st.session_state["workflow_text"] = st.session_state.get("latest_transcript", "")

    clone_allowed_by_platform = not IS_STREAMLIT_CLOUD or ALLOW_CLOUD_VOICE_CLONE
    clone_available = is_voice_clone_available() and clone_allowed_by_platform

    output_mode_options = [TTS_MODE_STANDARD, TTS_MODE_CLONE] if clone_available else [TTS_MODE_STANDARD]
    if is_speech_flow:
        output_mode_options = [OUTPUT_TEXT_ONLY, *output_mode_options]

    st.markdown("<p class='section-kicker'>Output</p>", unsafe_allow_html=True)
    default_output_mode = TTS_MODE_STANDARD
    _prepare_single_segment_value("workflow_output_mode", output_mode_options, default_output_mode)
    output_mode = st.segmented_control(
        "Output mode",
        output_mode_options,
        selection_mode="single",
        key="workflow_output_mode",
    )
    output_mode = _resolve_single_segment_value(
        "workflow_output_mode",
        output_mode_options,
        default_output_mode,
        output_mode,
    )

    if not clone_available:
        if IS_STREAMLIT_CLOUD and not ALLOW_CLOUD_VOICE_CLONE:
            st.info(
                "Voice Clone is disabled on Streamlit Cloud by default due RAM/latency limits. "
                "Set `ALLOW_CLOUD_VOICE_CLONE=1` to force-enable it."
            )
        else:
            st.info("Voice Clone is unavailable in this environment (XTTS dependency not installed).")

    needs_tts = output_mode != OUTPUT_TEXT_ONLY
    tts_mode = output_mode if needs_tts else TTS_MODE_STANDARD
    output_language_code = "tr"
    reference_media: MediaPayload | None = None
    keep_clone_model_loaded = False
    clone_terms_confirmed = True

    if needs_tts:
        st.markdown("<p class='section-kicker'>Speech Output</p>", unsafe_allow_html=True)
        selected_output_language = st.selectbox(
            "Language",
            list(TTS_LANGUAGES),
            key="workflow_tts_language",
        )
        output_language_code = TTS_LANGUAGES[selected_output_language]
        if tts_mode == TTS_MODE_CLONE:
            st.caption(f"Clone backend: {get_voice_clone_backend_name()}")
            if IS_STREAMLIT_CLOUD:
                st.info("Voice cloning on Streamlit Cloud can be slow and unstable due CPU/RAM limits.")
            _render_voice_clone_tips()
            st.markdown("<p class='section-kicker'>Import Reference Voice</p>", unsafe_allow_html=True)
            reference_media = render_media_source(
                title="Reference Voice",
                key_prefix="reference_media",
                helper_text=(
                    "Same module as source speech. Record or upload audio/video for cloning."
                ),
            )
            keep_clone_model_loaded = st.checkbox(
                "Keep clone model loaded in memory (faster, uses more RAM)",
                value=False,
                key="keep_clone_model_loaded",
            )
            clone_terms_confirmed = st.checkbox(
                "I confirm I have authorization to use this voice sample for AI-generated speech.",
                value=False,
                key="clone_terms_confirmed",
            )
    else:
        release_standard_tts_model()
        release_voice_clone_model()

    st.markdown("<p class='section-kicker'>Text To Preview</p>", unsafe_allow_html=True)
    st.text_area(
        "Text",
        key="workflow_text",
        height=190,
        help="Speech-to-text fills this automatically. You can edit before generation.",
    )

    st.markdown("<p class='section-kicker'>Run</p>", unsafe_allow_html=True)
    if is_speech_flow:
        if st.button("Run speech workflow", use_container_width=True):
            if source_media is None:
                st.warning("Please provide source speech first.")
            else:
                try:
                    transcription = _transcribe_source_media(
                        source_media=source_media,
                        language=transcription_language,
                        model_path=model_path,
                    )
                except Exception as exc:  # noqa: BLE001
                    _show_error("Transcription failed", exc)
                else:
                    st.success("Text extracted successfully.")
                    _render_transcription_summary(transcription)
                    if not needs_tts:
                        return
                    if tts_mode == TTS_MODE_CLONE and not clone_terms_confirmed:
                        st.warning("Please confirm voice usage rights before generating cloned speech.")
                        return
                    try:
                        result = _generate_tts_from_text(
                            text=st.session_state["workflow_text"],
                            tts_mode=tts_mode,
                            language_code=output_language_code,
                            reference_media=reference_media,
                            keep_clone_model_loaded=keep_clone_model_loaded,
                        )
                    except Exception as exc:  # noqa: BLE001
                        _show_error("Speech generation failed", exc)
                    else:
                        _render_tts_result(result)
    else:
        if st.button("Generate speech from text", use_container_width=True):
            if tts_mode == TTS_MODE_CLONE and not clone_terms_confirmed:
                st.warning("Please confirm voice usage rights before generating cloned speech.")
                return
            try:
                result = _generate_tts_from_text(
                    text=st.session_state["workflow_text"],
                    tts_mode=tts_mode,
                    language_code=output_language_code,
                    reference_media=reference_media,
                    keep_clone_model_loaded=keep_clone_model_loaded,
                )
            except Exception as exc:  # noqa: BLE001
                _show_error("Speech generation failed", exc)
            else:
                _render_tts_result(result)


if __name__ == "__main__":
    main()
