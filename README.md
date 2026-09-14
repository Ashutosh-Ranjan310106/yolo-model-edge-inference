# YOLO Model Edge Inference & Mobile CDN 🚀

Mobile-optimized ONNX models and standalone edge-inference web client for real-time object detection and monocular depth estimation.

Designed for real-time obstacle detection and assistive navigation on mobile devices (Smartphones, Wearables, PWA).

---

## 📦 Distributable ONNX Models & Global CDN Links

All models in this repository are publicly accessible worldwide via high-speed CDNs:

| Model ID | Task / Description | Resolution | Size | Direct Raw URL / CDN Link |
|---|---|---|---|---|
| **`yolo26n-nav-run6_320`** | YOLO26 Nano (Best 46 Classes) | 320×320 | 9.26 MB | [Download](https://raw.githubusercontent.com/Ashutosh-Ranjan310106/yolo-model-edge-inference/main/models/yolo26n-nav-run6_320.onnx) \| [jsDelivr CDN](https://cdn.jsdelivr.net/gh/Ashutosh-Ranjan310106/yolo-model-edge-inference@main/models/yolo26n-nav-run6_320.onnx) |
| **`yolo26n-nav-run6_480`** | YOLO26 Nano (Best 46 Classes) | 480×480 | 9.31 MB | [Download](https://raw.githubusercontent.com/Ashutosh-Ranjan310106/yolo-model-edge-inference/main/models/yolo26n-nav-run6_480.onnx) \| [jsDelivr CDN](https://cdn.jsdelivr.net/gh/Ashutosh-Ranjan310106/yolo-model-edge-inference@main/models/yolo26n-nav-run6_480.onnx) |
| **`yolo26n-24k-nav-run1_320`** | YOLO26 Nano (24k Dataset) | 320×320 | 9.24 MB | [Download](https://raw.githubusercontent.com/Ashutosh-Ranjan310106/yolo-model-edge-inference/main/models/yolo26n-24k-nav-run1_best_320.onnx) \| [jsDelivr CDN](https://cdn.jsdelivr.net/gh/Ashutosh-Ranjan310106/yolo-model-edge-inference@main/models/yolo26n-24k-nav-run1_best_320.onnx) |
| **`depth_anything_v2_small`** | Depth Anything V2 (INT8 Quantized) | 518×518 | 26.0 MB | [Download](https://raw.githubusercontent.com/Ashutosh-Ranjan310106/yolo-model-edge-inference/main/models/depth_anything_v2_small_quantized.onnx) \| [jsDelivr CDN](https://cdn.jsdelivr.net/gh/Ashutosh-Ranjan310106/yolo-model-edge-inference@main/models/depth_anything_v2_small_quantized.onnx) |
| **`yolo26n_depth`** | YOLO26 Nano Metric Depth | 768×768 | 19.81 MB | [Download](https://raw.githubusercontent.com/Ashutosh-Ranjan310106/yolo-model-edge-inference/main/models/yolo26n_depth.onnx) \| [jsDelivr CDN](https://cdn.jsdelivr.net/gh/Ashutosh-Ranjan310106/yolo-model-edge-inference@main/models/yolo26n_depth.onnx) |
| **`yolo26s_depth`** | YOLO26 Small Metric Depth | 768×768 | 45.98 MB | [Download](https://raw.githubusercontent.com/Ashutosh-Ranjan310106/yolo-model-edge-inference/main/models/yolo26s_depth.onnx) \| [jsDelivr CDN](https://cdn.jsdelivr.net/gh/Ashutosh-Ranjan310106/yolo-model-edge-inference@main/models/yolo26s_depth.onnx) |

Full metadata and checksums are recorded in [`models/models_manifest.json`](./models/models_manifest.json).

---

## 🎯 Custom Navigation Classes (46 Classes)
`yolo26n-nav-run6` detects the following 46 navigation & obstacle classes:
1. `Bike` 2. `Building` 3. `Car` 4. `Person` 5. `Stairs` 6. `Traffic sign` 7. `Electrical Pole` 8. `Road` 9. `Motorcycle` 10. `Dustbin` 11. `Dog` 12. `Manhole` 13. `Tree` 14. `Guard rail` 15. `Pedestrian crosswalk` 16. `Truck` 17. `Bus` 18. `Bench` 19. `Traffic Cone` 20. `Fire hydrant` 21. `Teraffic Barrel` 22. `Plant Pot` 23. `Electrical Box` 24. `Chair` 25. `Bicycle Rack` 26. `Door` 27. `Wall` 28. `Bus station` 29. `Barricade` 30. `Table` 31. `Desk` 32. `Bookshelf/Storage` 33. `Window` 34. `Sink` 35. `Toilet` 36. `Sign_Board` 37. `Elevator` 38. `Train` 39. `Cat` 40. `Traffic Light` 41. `Ramp` 42. `Escalator` 43. `Animal` 44. `Computer` 45. `Drawer` 46. `TV`

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
