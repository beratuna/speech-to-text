# Speech to Text

Simple Streamlit app for transcribing Turkish and English audio.

Backend is selected automatically:
- Apple Silicon macOS: `mlx-whisper`
- Other platforms (including Streamlit Cloud): `faster-whisper`

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
