# ROD Edge Inference — Mobile Web Client

Mobile-first Progressive Web App (PWA) that runs YOLO26 inference **completely on-device** directly inside mobile Chrome/Firefox on an Android phone.

---

## How It Works

1. **Zero Native Install**: Open `http://<LAN-IP>:8000/client/` on your Android phone browser.
2. **Local Model Caching**: Downloads the `.onnx` model from the backend once, saves it inside browser IndexedDB, and operates 100% offline subsequently.
3. **Latest-Frame Camera Strategy**: Captures frames from the phone's environment camera using `navigator.mediaDevices.getUserMedia`. Drops intermediate frames to guarantee zero buffering latency.
4. **On-Device Execution**: Runs model inference on the phone using ONNX Runtime Web (WASM SIMD or WebGPU acceleration).
5. **Real-Time HUD**: Renders bounding boxes, class labels, confidence scores, estimated distances, actual measured FPS, and per-stage latency breakdown (preprocess, inference, postprocess, total ms).
