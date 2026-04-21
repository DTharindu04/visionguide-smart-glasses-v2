# Environment Structure

This project is offline-first after setup. Runtime configuration is read from `config/runtime.toml`, with selected overrides from environment variables using the `SMART_GLASSES_` prefix. The `.env.example` file documents supported overrides, but production services should inject variables through systemd, shell profile, or a supervisor.

## Required Local Directories

```text
models/
  manifest.local.toml
  object_detection/
    yolov8n.onnx
    coco_labels.txt
  face_detection/
    face_detection_yunet_2023mar.onnx
  face_recognition/
    face_recognition_sface_2021dec.onnx
  emotion/
    emotion-ferplus-8.onnx
  ocr/
    tessdata/
      eng.traineddata
      sin.traineddata

storage/
  faces/
  cache/
  logs/
  settings/
```

## Profiles

- `pi4_conservative`: lower FPS and heat for long runtime.
- `pi4_balanced`: default field profile for Pi 4.
- `pi4_responsive`: faster safety inference, intended for good cooling and stable power.

Mode-specific performance profiles are capped by the selected Pi profile and camera settings. For example, `DANGER` may request higher responsiveness, but it will not exceed the selected camera target FPS.

## System Packages

Install Pi-specific packages from Raspberry Pi OS where possible:

```bash
sudo apt install -y python3-picamera2 python3-opencv libcamera-apps tesseract-ocr espeak-ng portaudio19-dev libatlas-base-dev
```

Raspberry Pi OS Bookworm manages system Python externally. Create the project virtual environment with access to apt-provided packages:

```bash
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Python dependencies are split across:

- `requirements/pi.txt`: Pi runtime venv dependencies, including `tflite-runtime`.
- `requirements/base.txt`: shared pure Python dependencies.
- `requirements/dev.txt`: workstation and CI-only dependencies, including pip OpenCV.

## Local Data Policy

- Face embeddings live under `storage/faces/`.
- OCR cache lives under `storage/cache/`.
- Runtime logs live under `storage/logs/`.
- Model files stay under `models/` and are not committed to git.
- Installed model provenance lives in `models/manifest.local.toml` and must include version, license, SHA-256 hash, input schema, and output schema for every installed model asset.
- The application must remain functional without network access after setup.

## Field Tuning

Thresholds and cooldowns are configured in `config/runtime.toml`. Tune these values during field testing instead of editing Python:

- object confidence, danger-zone, closeness, and severity thresholds
- face quality and recognition thresholds
- OCR confidence, cache, and preprocessing settings
- speech cooldowns and stale-message TTLs
