# Speech to Text

Minimal Streamlit app for Turkish/English speech workflows:
- Speech-to-text (audio/video upload + mic recording)
- Text-to-speech (TR/EN)

<!-- LOC_START -->Python LOC: `737` across `8` files (auto-updated).<!-- LOC_END -->

Backend is selected automatically:
- Apple Silicon macOS: `mlx-whisper`
- Other platforms (including Streamlit Cloud): `faster-whisper`

TTS backend is also selected automatically:
- Apple Silicon macOS: `mlx-audio` (`mlx-community/chatterbox-6bit`) when installed
- Other platforms (including Streamlit Cloud): `chatterbox-tts` (CPU) when available
- Fallback: `gTTS` when `chatterbox-tts` is unavailable/incompatible

Model availability:
- Apple Silicon: Large/Medium/Small
- Non-Apple and Streamlit Cloud: Small/Medium (Large is hidden to avoid deploy/runtime failures)

TTS notes:
- Standard mode keeps the existing auto backend selection (`mlx-audio` / `chatterbox-tts` / `gTTS`)
- Voice Clone mode uses Coqui XTTS v2 (zero-shot, local-only)
- Very long text is allowed with a soft warning (chunk-and-concat is planned)

## TTS v2: Voice Clone

Voice cloning is available in **Text-to-Speech → Voice Clone** mode:
- Upload a 3-30s reference voice clip (WAV, MP3, M4A, OGG, FLAC)
- The app preprocesses the clip (mono, 22_050 Hz, silence trim)
- XTTS v2 synthesizes Turkish or English speech in the reference voice
- XTTS model weights are downloaded once and cached
- XTTS dependency (`TTS>=0.22.0`) currently supports Python `<3.12`, so Voice Clone is hidden when unavailable

Expected short-text inference speed:

| Platform | Backend | Speed |
|---|---|---|
| Apple Silicon macOS | XTTS v2 (MPS) | ~3-8s |
| Local CPU | XTTS v2 (CPU) | ~10-20s |
| Streamlit Cloud | XTTS v2 (CPU) | ~20-40s |

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
