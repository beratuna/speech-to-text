# Speech to Text

Minimal Streamlit app for Turkish/English speech workflows:
- Speech-to-text (audio/video upload + mic recording)
- Text-to-speech (TR/EN)

Backend is selected automatically:
- Apple Silicon macOS: `mlx-whisper`
- Other platforms (including Streamlit Cloud): `faster-whisper`

TTS backend is also selected automatically:
- Apple Silicon macOS: `mlx-audio` (`mlx-community/chatterbox-6bit`) when installed
- Other platforms (including Streamlit Cloud): `chatterbox-tts` (CPU)

Model availability:
- Apple Silicon: Large/Medium/Small
- Non-Apple and Streamlit Cloud: Small/Medium (Large is hidden to avoid deploy/runtime failures)

TTS v1 notes:
- Single default multilingual voice/model (minimal UI)
- No cloud fallback backend yet
- Very long text is allowed with a soft warning (chunk-and-concat is planned)

## Run Locally

```bash
uv sync --dev
uv run streamlit run app.py
```

## Share via Streamlit Cloud

1. Push this repo to GitHub.
2. In Streamlit Community Cloud, create a new app from this repo.
3. Set the main file path to `app.py`.
4. Share the generated app URL.
