# Speech to Text

Simple Streamlit app for transcribing Turkish and English audio with `mlx-whisper`.

## Run

```bash
uv sync --dev
uv run streamlit run app.py
```

## Share With Non-Technical Users (macOS)

1. Send this project folder as a zip.
2. Ask them to unzip and double-click `Speech to Text.app`.
3. The launcher opens Terminal, starts the app server, and opens the browser.

Notes:
- First run can take a while because model files are downloaded.
- They need `uv` installed once on their Mac.

## How This Becomes a `.app`

`Speech to Text.app` is a macOS launcher app that runs `run_app.sh`.
That script:
1. Ensures `uv` exists.
2. Creates `.venv` on first launch with `uv sync --dev`.
3. Starts Streamlit and opens the browser.

If you ever need to rebuild the app, run:

```bash
osacompile -l JavaScript -o "Speech to Text.app" -e 'ObjC.import("Foundation");function q(s){return "'"'"'" + s.replace(/'"'"'/g,"'"'"'\"'"'"'\"'"'"'") + "'"'"'";}function run(){var t=Application("Terminal");t.activate();var b=ObjC.unwrap($.NSBundle.mainBundle.bundlePath);var r=ObjC.unwrap($(b).stringByDeletingLastPathComponent);t.doScript(q(r + "/run_app.sh"));}'
```
