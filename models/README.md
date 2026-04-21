# Model Assets

Model files are stored locally and are intentionally not committed to git.

Expected runtime paths are defined in `config/settings.py`:

```text
models/manifest.local.toml
models/object_detection/yolov8n.onnx
models/object_detection/coco_labels.txt
models/face_detection/face_detection_yunet_2023mar.onnx
models/face_recognition/face_recognition_sface_2021dec.onnx
models/emotion/emotion-ferplus-8.onnx
models/ocr/tessdata/eng.traineddata
models/ocr/tessdata/sin.traineddata
```

The safety runtime requires the object detector and labels. Face, OCR, and emotion assets are separate feature tiers so the system can degrade gracefully when non-safety assets are unavailable. Sinhala OCR data is included in the path contract for future multilingual support.

Copy `models/manifest.example.toml` to `models/manifest.local.toml` after installing real assets, then update any version, license, SHA-256 hash, input schema, and output schema values that differ from the installed files. The validator checks hashes for installed files.

Commercial deployment note: the included YOLOv8n manifest entry is AGPL-3.0. Replace it with a commercially licensed detector or obtain the required commercial license before shipping a closed-source product.
