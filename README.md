# VisionX: Edge AI Perception & Assistive Intelligence Platform 🚀

[![ONNX Runtime Web](https://img.shields.io/badge/ONNX_Runtime_Web-1.21.0-blue.svg)](https://onnxruntime.ai/)
[![Vite](https://img.shields.io/badge/Vite-6.4.3-purple.svg)](https://vitejs.dev/)
[![React](https://img.shields.io/badge/React-18.3.1-61dafb.svg)](https://react.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**VisionX** is a next-generation multimodal assistive intelligence ecosystem uniting **100% on-device visual edge navigation** with **cloud-accelerated document synthesis**.

- 🧭 **VisionX Navigation**: Fully client-side obstacle detection, 3D metric depth perception, walking corridor collision avoidance, and directional auditory guidance running directly in the browser via ONNX Runtime Web.
- 📝 **Handwritten Notes Analyser**: AI-powered document digitization, mathematical formula OCR, and concept synthesis ([Live App](https://handwritten-notes-analyser-sgwn.vercel.app/)).

---

## 📦 Distributable ONNX Models & High-Speed CDN Endpoints

All models are hosted with public CORS headers (`Access-Control-Allow-Origin: *`) and download directly into client-side browser cache:

### 1. Object Detection Models
| Model ID | User-Facing Name | Resolution | Classes | Size | Direct CDN URL |
|---|---|---|---|---|---|
| **`yolo26n_nav_run8_256`** | YOLO26-Nano Fast (Run 8) | 256×256 | 25 | 9.22 MB | [Download](https://raw.githubusercontent.com/Ashutosh-Ranjan310106/yolo-model-edge-inference/main/models/yolo26n_nav_run8_256.onnx) |
| **`yolo26s_nav_run683_416`** | YOLO26-Small Balanced (Run 683) | 416×416 | 25 | 36.33 MB | [Download](https://raw.githubusercontent.com/Ashutosh-Ranjan310106/yolo-model-edge-inference/main/models/yolo26s_nav_run683_416.onnx) |

### 2. Dense Metric & Relative Depth Models
| Model ID | User-Facing Name | Resolution | Type | Size | Direct CDN URL |
|---|---|---|---|---|---|
| **`depth_256`** | YOLO26-Nano Depth | 256×256 | Metric (m) | 19.81 MB | [Download](https://raw.githubusercontent.com/Ashutosh-Ranjan310106/yolo-model-edge-inference/main/models/depth/depth_256.onnx) |
| **`depth_320`** | YOLO26-Nano Depth | 320×320 | Metric (m) | 19.81 MB | [Download](https://raw.githubusercontent.com/Ashutosh-Ranjan310106/yolo-model-edge-inference/main/models/depth/depth_320.onnx) |
| **`depth_512`** | YOLO26-Nano Depth | 512×512 | Metric (m) | 19.81 MB | [Download](https://raw.githubusercontent.com/Ashutosh-Ranjan310106/yolo-model-edge-inference/main/models/depth/depth_512.onnx) |
| **`yolo26s_depth_256`** | YOLO26-Small Depth | 256×256 | Metric (m) | 45.98 MB | [Download](https://raw.githubusercontent.com/Ashutosh-Ranjan310106/yolo-model-edge-inference/main/models/depth/yolo26s_depth_256.onnx) |
| **`yolo26s_depth_320`** | YOLO26-Small Depth | 320×320 | Metric (m) | 45.98 MB | [Download](https://raw.githubusercontent.com/Ashutosh-Ranjan310106/yolo-model-edge-inference/main/models/depth/yolo26s_depth_320.onnx) |
| **`yolo26m_depth_256`** | YOLO26-Medium Depth | 256×256 | Metric (m) | 84.33 MB | [Download](https://raw.githubusercontent.com/Ashutosh-Ranjan310106/yolo-model-edge-inference/main/models/depth/yolo26m_depth_256.onnx) |
| **`yolo26m_depth_320`** | YOLO26-Medium Depth | 320×320 | Metric (m) | 84.33 MB | [Download](https://raw.githubusercontent.com/Ashutosh-Ranjan310106/yolo-model-edge-inference/main/models/depth/yolo26m_depth_320.onnx) |
| **`depth_anything_v2_small_quantized`** | Depth Anything V2 Small | 518×518 | Relative | 26.00 MB | [Download](https://raw.githubusercontent.com/Ashutosh-Ranjan310106/yolo-model-edge-inference/main/models/depth_anything_v2_small_quantized.onnx) |

Full metadata, input/output tensor shapes, and checksums are maintained in [`models/models_manifest.json`](./models/models_manifest.json).

---

## 🎯 Active Navigation Classes (25 Classes)
Both `yolo26n_nav_run8` and `yolo26s_nav_run683` share the unified 25-class navigation vocabulary:
1. `Bike/Motorcycle` 2. `Building` 3. `Vehicle` 4. `Person` 5. `Stairs` 6. `Traffic sign` 7. `Electrical Pole` 8. `Dustbin` 9. `Animal` 10. `Manhole` 11. `Tree` 12. `Guard rail` 13. `Pedestrian crosswalk` 14. `Bench` 15. `Traffic Cone` 16. `Teraffic Barrel` 17. `Plant Pot` 18. `Chair` 19. `Door` 20. `Table/Desk` 21. `Bookshelf/Storage` 22. `Window` 23. `Sign_Board` 24. `Display` 25. `Drawer`

---

## ⚡ Quickstart: Running VisionX App

### 1. VisionX Standalone React Web Client (`frontend/`)
The complete modern VisionX React app is located in `frontend/`.

```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` on desktop or phone.

#### Production Build:
```bash
cd frontend
npm run build
```
The compiled, production-ready static assets are in `frontend/dist/`. You can serve them with any static web server:
```bash
python -m http.server -d frontend/dist 8080
```

### 2. Optional: FastAPI Registry & Offline Weights Backend (`backend/`)
```bash
cd backend
pip install -r requirements.txt
python -m app.main
```
FastAPI runs on `http://0.0.0.0:8000`.

---

## 📱 Platform Features & Architecture
- **100% Client-Side Edge Inference**: Video frames never leave RAM. Zero cloud dependencies for perception.
- **Unified VisionX Portal**: Seamless switching between on-device Navigation and Handwritten Notes Analyser.
- **Dual-Model Real-Time Fusion**: YOLO26 object detection (~15 FPS) fused with Dense Depth estimation (~6-10 FPS).
- **Spatial Audio Guidance**: 7-zone direction tracking, path sector evaluation, and obstacle avoidance alerts.
- **WASM SIMD Acceleration**: Single-threaded, zero-worker-restriction ONNX Runtime Web engine.
- **Offline Cache**: Browser CacheStorage / IndexedDB caches models after initial download for zero-bandwidth offline usage.
