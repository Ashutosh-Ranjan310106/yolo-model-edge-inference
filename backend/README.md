# ROD Edge Inference — Backend Model Registry

A lightweight, dedicated model registry and artifact distribution service for on-device ROD YOLO inference.

> **CRITICAL ARCHITECTURAL GUARANTEE**:
> This backend **NEVER** performs inference. It does not receive camera frames or video streams.
> Its exclusive role is model discovery, metadata distribution, versioning, SHA-256 integrity verification, and binary model streaming.

---

## Features

- **FastAPI Core**: High-throughput asynchronous REST API.
- **Model Discovery**: Automatically scans `weights/` for `.onnx`, NCNN archives, and `.pt` checkpoints.
- **Checksum Verification**: Provides SHA-256 checksums to guarantee on-device model integrity.
- **Streaming Downloads**: Supports HTTP Range headers and chunked delivery with client-side progress bars.
- **Companion Metadata**: Emits class names (27 navigation classes), tensor input/output shapes, normalization parameters, and distance heuristic tiers.
- **Integrated Client Static Server**: Serves the mobile web edge detector for instant LAN testing.

---

## Directory Layout

```
backend/
├── app/
│   ├── main.py             # FastAPI entrypoint and static client mounting
│   ├── config/             # Path configuration and settings
│   ├── schemas/            # Pydantic data contracts
│   ├── services/           # Model registry and scanning logic
│   └── api/                # /api/health and /api/models routers
├── weights/                # Stored mobile models (.onnx, ncnn, .pt)
├── export_models.py        # CLI export utility from PyTorch weights
└── requirements.txt        # Backend dependencies
```

---

## Setup & Running

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Export Models** (if adding new `.pt` weights):
   ```bash
   python export_models.py --weights path/to/best.pt --resolutions 480 320 --formats onnx ncnn
   ```

3. **Start Server**:
   ```bash
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

4. **Verify Health**:
   ```bash
   curl http://localhost:8000/api/health
   ```
