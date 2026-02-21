# Speech to Text

Simple Streamlit app for transcribing Turkish and English audio.

Backend is selected automatically:
- Apple Silicon macOS: `mlx-whisper`
- Other platforms (including Streamlit Cloud): `faster-whisper`

Model availability:
- Apple Silicon: Large/Medium/Small
- Non-Apple and Streamlit Cloud: Small/Medium (Large is hidden to avoid deploy/runtime failures)

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
