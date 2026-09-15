# YOLO Model Edge Inference & Mobile CDN 🚀

Mobile-optimized ONNX models and standalone edge-inference web client for real-time object detection and monocular depth estimation.

Designed for real-time obstacle detection and assistive navigation on mobile devices (Smartphones, Wearables, PWA).

---

## 📦 Distributable ONNX Models & Global CDN Links

All models in this repository are publicly accessible worldwide via high-speed CDNs:

| Model ID | User-Facing Name | Resolution | Classes | Size | Direct Raw URL / CDN Link |
|---|---|---|---|---|---|
| **`yolo26n_nav_run1_320`** | YOLO26-Nano (Run 1) | 320×320 | 27 | 9.24 MB | [Download](https://raw.githubusercontent.com/Ashutosh-Ranjan310106/yolo-model-edge-inference/main/models/yolo26n_nav_run1_320.onnx) |
| **`yolo26n_nav_run1_480`** | YOLO26-Nano (Run 1) | 480×480 | 27 | 9.30 MB | [Download](https://raw.githubusercontent.com/Ashutosh-Ranjan310106/yolo-model-edge-inference/main/models/yolo26n_nav_run1_480.onnx) |
| **`yolo26n_nav_run8_256`** | YOLO26-Nano Fast (Run 8) | 256×256 | 25 | 9.22 MB | [Download](https://raw.githubusercontent.com/Ashutosh-Ranjan310106/yolo-model-edge-inference/main/models/yolo26n_nav_run8_256.onnx) |
| **`yolo26s_nav_run291_320`** | YOLO26-Small Accurate (Run 291) | 320×320 | 27 | 36.32 MB | [Download](https://raw.githubusercontent.com/Ashutosh-Ranjan310106/yolo-model-edge-inference/main/models/yolo26s_nav_run291_320.onnx) |
| **`yolo26s_nav_run291_480`** | YOLO26-Small Accurate (Run 291) | 480×480 | 27 | 36.37 MB | [Download](https://raw.githubusercontent.com/Ashutosh-Ranjan310106/yolo-model-edge-inference/main/models/yolo26s_nav_run291_480.onnx) |
| **`yolo26s_nav_run683_416`** | YOLO26-Small Balanced (Run 683) | 416×416 | 25 | 36.33 MB | [Download](https://raw.githubusercontent.com/Ashutosh-Ranjan310106/yolo-model-edge-inference/main/models/yolo26s_nav_run683_416.onnx) |
| **`depth_256`** | YOLO26-Depth Metric (Low Power) | 256×256 | Dense | 19.81 MB | [Download](https://raw.githubusercontent.com/Ashutosh-Ranjan310106/yolo-model-edge-inference/main/models/depth/depth_256.onnx) |
| **`depth_320`** | YOLO26-Depth Metric (Fast) | 320×320 | Dense | 19.81 MB | [Download](https://raw.githubusercontent.com/Ashutosh-Ranjan310106/yolo-model-edge-inference/main/models/depth/depth_320.onnx) |
| **`depth_512`** | YOLO26-Depth Metric (Quality) | 512×512 | Dense | 19.81 MB | [Download](https://raw.githubusercontent.com/Ashutosh-Ranjan310106/yolo-model-edge-inference/main/models/depth/depth_512.onnx) |
| **`depth_anything_v2_small`** | Depth Anything V2 (Quantized) | 518×518 | Relative | 26.0 MB | [Download](https://raw.githubusercontent.com/Ashutosh-Ranjan310106/yolo-model-edge-inference/main/models/depth_anything_v2_small_quantized.onnx) |

Full metadata and checksums are recorded in [`models/models_manifest.json`](./models/models_manifest.json).

---

## 🎯 Supported Class Sets
- **27 Classes (`run1` & `run291`)**: Bike, Building, Car, Person, Stairs, Traffic sign, Electrical Pole, Road, Motorcycle, Dustbin, Dog, Manhole, Tree, Guard rail, Pedestrian crosswalk, Truck, Bus, Bench, Traffic Cone, Fire hydrant, Teraffic Barrel, Plant Pot, Electrical Box, Chair, Bicycle Rack, Door, Wall.
- **25 Classes (`run8` & `run683`)**: Bike/Motorcycle, Building, Vehicle, Person, Stairs, Traffic sign, Electrical Pole, Dustbin, Animal, Manhole, Tree, Guard rail, Pedestrian crosswalk, Bench, Traffic Cone, Teraffic Barrel, Plant Pot, Chair, Door, Table/Desk, Bookshelf/Storage, Window, Sign_Board, Display, Drawer.

---

## ⚡ Quickstart

### 1. Launch the Edge Inference Client
Open `frontend/index.html` in any browser or serve it with Python:
```bash
cd frontend
python -m http.server 8080
```
Visit `http://localhost:8080` on desktop or mobile.

### 2. Optional: Run the FastAPI Registry Backend
```bash
cd backend
pip install -r requirements.txt
python -m app.main
```
Server starts on `http://0.0.0.0:8000`.

---

## 📱 Features
- **Zero-Install PWA**: Runs directly in Safari, Chrome, Edge, and Firefox.
- **ONNX Runtime Web (WASM SIMD)**: Single-threaded SIMD acceleration with zero cross-origin worker restrictions.
- **Offline-First Storage**: Models are cached into browser `IndexedDB` on the first run; subsequent runs require zero internet connection.
- **Dual Inference**: Runs custom YOLO detection and Depth Anything V2 concurrently with temporal tracking and obstacle collision warnings.
