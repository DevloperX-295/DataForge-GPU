# DataForge GPU – AI Training Data Debugger

A free, Python-powered local tool for inspecting, validating, cleaning, and analyzing datasets before AI model training. The app includes a browser-based interface and optional Hugging Face model loading.

## Requirements

- Windows 10/11
- Python 3.10 or newer
- Internet connection for installing Python packages and downloading selected Hugging Face models
- Optional: a compatible NVIDIA GPU and correctly configured PyTorch/CUDA installation for GPU acceleration

## Quick start (Windows)

1. Install Python 3.10+ from https://www.python.org/downloads/. During setup, enable **Add Python to PATH**.
2. Download this repository (GitHub **Code → Download ZIP**) and extract it.
3. Double-click `run.bat`.
4. Open http://127.0.0.1:8000 if your browser does not open automatically.

The launcher installs the dependencies listed in `requirements.txt` and starts the app locally.

## Manual start

```bash
python -m pip install -r requirements.txt
python app.py
```

Then visit http://127.0.0.1:8000.

## Important security and usage notes

- This project is designed to run on your own computer. The server binds to `127.0.0.1` (localhost); do not expose it to the public internet, a network interface, or a port-forward without adding authentication, request limits, and security protections.
- URL ingestion fetches user-supplied URLs. Only submit URLs you trust. Do not use this app as a public multi-user service in its current form.
- Loading Hugging Face models can download large files and consume substantial RAM, VRAM, and disk space. Only load model IDs you trust and that your hardware can support.
- The current session dataset is held in memory; export important results before closing the app.
- GPU acceleration depends on the local PyTorch/CUDA setup. CPU use may be possible but slower.

## Project files

- `app.py` – FastAPI backend and browser UI
- `run.bat` – Windows launcher; checks for Python 3.10+ and installs requirements
- `requirements.txt` – Python dependencies

## License

No license has been specified yet. All rights remain with the project author unless a license is added.
