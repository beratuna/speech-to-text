from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

SUPPORTED_AUDIO_TYPES = ["wav", "mp3", "m4a", "ogg", "flac"]
SUPPORTED_VIDEO_TYPES = ["mp4", "mov", "m4v", "mkv", "webm", "avi"]
SUPPORTED_MEDIA_TYPES = [*SUPPORTED_AUDIO_TYPES, *SUPPORTED_VIDEO_TYPES]
SOFT_WARNING_BYTES = 100 * 1024 * 1024


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


@dataclass(frozen=True)
class MediaPayload:
    name: str
    content: bytes
    suffix: str

    @property
    def size(self) -> int:
        return len(self.content)

    @property
    def is_video(self) -> bool:
        return self.suffix.lower().lstrip(".") in SUPPORTED_VIDEO_TYPES


def _to_payload(source, fallback_name: str) -> MediaPayload | None:
    if source is None:
        return None
    name = getattr(source, "name", fallback_name)
    suffix = Path(name).suffix.lower() or ".wav"
    return MediaPayload(name=name, content=source.getvalue(), suffix=suffix)


def _preview_payload(payload: MediaPayload) -> None:
    if payload.is_video:
        st.video(payload.content)
    else:
        st.audio(payload.content)


def render_media_source(title: str, key_prefix: str, helper_text: str | None = None) -> MediaPayload | None:
    st.subheader(title)
    if helper_text:
        st.markdown(f"<p class='ui-muted'>{helper_text}</p>", unsafe_allow_html=True)

    upload_col, divider_col, record_col = st.columns([6, 1, 4], gap="small")
    with upload_col:
        st.markdown(
            (
                "<div class='media-option-card'>"
                "<div class='media-option-icon'>📁</div>"
                "<div class='media-option-title'>Add or drop a file</div>"
                "<div class='media-option-caption'>Audio/video formats are supported</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            f"Upload {title.lower()}",
            type=SUPPORTED_MEDIA_TYPES,
            key=f"{key_prefix}_upload",
            help="Supported: wav, mp3, m4a, ogg, flac, mp4, mov, m4v, mkv, webm, avi",
        )

    with divider_col:
        st.markdown("<div class='media-option-divider'>or</div>", unsafe_allow_html=True)

    with record_col:
        st.markdown(
            (
                "<div class='media-option-card'>"
                "<div class='media-option-icon'>🎙️</div>"
                "<div class='media-option-title'>Record audio</div>"
                "<div class='media-option-caption'>Use your mic in-browser</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        recorded = st.audio_input(
            f"Record {title.lower()}",
            key=f"{key_prefix}_record",
            help=(
                "Click the mic icon to start/stop recording. "
                "Recording does not auto-stop on silence."
            ),
        )

    upload_payload = _to_payload(uploaded, fallback_name=f"{key_prefix}.wav")
    record_payload = _to_payload(recorded, fallback_name=f"{key_prefix}.wav")

    payload = upload_payload or record_payload
    if upload_payload and record_payload:
        active_source_options = ["Uploaded file", "Recording"]
        _prepare_single_segment_value(
            key=f"{key_prefix}_active_source",
            options=active_source_options,
            default="Uploaded file",
        )
        active_source = st.segmented_control(
            "Use sample from",
            active_source_options,
            selection_mode="single",
            key=f"{key_prefix}_active_source",
        )
        active_source = _resolve_single_segment_value(
            key=f"{key_prefix}_active_source",
            options=active_source_options,
            default="Uploaded file",
            value=active_source,
        )
        payload = upload_payload if active_source == "Uploaded file" else record_payload

    if payload is None:
        return None

    st.markdown("**Uploaded Samples** 1/1")
    _preview_payload(payload)
    if payload.size > SOFT_WARNING_BYTES:
        st.warning("Large media detected (>100MB). Processing may take longer.")
    st.markdown(
        (
            "<div class='media-guidance-box'>"
            "Use a clean voice sample in a quiet environment. "
            "Reverb, multiple speakers, and background noise reduce quality."
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    return payload


def save_media_payload(payload: MediaPayload) -> Path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=payload.suffix or ".tmp") as temp_file:
        temp_file.write(payload.content)
        return Path(temp_file.name)


def _extract_audio_from_video(video_path: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("`ffmpeg` is required to use video files as voice reference input.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        output_path = Path(temp_file.name)

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "22050",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        output_path.unlink(missing_ok=True)
        details = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError(f"Failed to extract audio from video: {details}") from exc
    return output_path


def to_reference_audio_payload(payload: MediaPayload) -> tuple[bytes, str]:
    if not payload.is_video:
        return payload.content, payload.suffix

    video_path = save_media_payload(payload)
    audio_path: Path | None = None
    try:
        audio_path = _extract_audio_from_video(video_path)
        return audio_path.read_bytes(), ".wav"
    finally:
        video_path.unlink(missing_ok=True)
        if audio_path is not None:
            audio_path.unlink(missing_ok=True)
