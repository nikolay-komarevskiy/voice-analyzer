![Voice Analyzer UI](UI.png)

### Running Locally

1. Install Python deps (consider a virtualenv):
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the app,  which serves backend and the static UI:
   ```bash
   python -m backend.main
   ```
3. Browse to `http://localhost:8000`

### Native Dependencies

Some audio backends require system libraries or developer headers to be present before Python deps can install cleanly.

- **macOS**: Install PortAudio via Homebrew (`brew install portaudio`) to satisfy the PyAudio backend. CoreAudio permissions must allow microphone access for the terminal session running the server.
- **Linux**: Install ALSA and PortAudio development packages (e.g. `sudo apt install libasound2-dev portaudio19-dev`) plus `python3-dev` so both PyAudio and sounddevice can compile against your distro’s audio stack.

If you only plan to use the synthetic backend, these native packages aren’t strictly required, but the real audio backends will fail to initialize without them.
