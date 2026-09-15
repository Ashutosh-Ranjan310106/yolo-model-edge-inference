# VisionX Navigation System — Pure React Standalone App

A standalone, mobile-first visual obstacle detection and navigation assistance web application running **100% on-device in the browser** using ONNX Runtime Web (WebAssembly SIMD / WebGPU).

**No backend server required. Zero Python dependencies.**

---

## 1. Key Features

- **100% Client-Side Edge Inference**: Neural networks run directly on your phone, tablet, or PC using ONNX Runtime Web. Camera frames never leave device RAM.
- **Dual-Model Perception Pipeline**:
  - **YOLO26 Object Detector (~15 FPS)**: Real-time detection across 27 obstacle classes (Person, Stairs, Car, Bike, Manhole, Traffic Sign, etc.).
  - **Dense Depth Estimator (~6 FPS)**: Metric depth estimation (YOLO-Depth) or relative depth (Depth Anything V2) with colormap overlay.
- **3D Spatial Fusion**:
  - Central ROI median metric depth extraction.
  - 7-zone continuous direction tracking (`far left`, `left`, `slightly left`, `center`, `slightly right`, `right`, `far right`).
  - 3-sector path clearance (`Left`, `Center`, `Right`).
  - Walking corridor trapezoid overlap collision detection.
- **Temporal Tracking & Smoothing**: Prevents flicker and smooths bounding boxes and distances across frames.
- **Voice Guidance**: Real-time audio alerts via the browser Web Speech Synthesis API.
- **Offline Capable (IndexedDB Storage)**: Models are downloaded once from CDN or loaded from disk, cached in browser IndexedDB, and run fully offline without internet connectivity.
- **Direct Local File Picker**: Easily load custom `.onnx` model files directly from your computer/phone storage.

---

## 2. Quick Start

### Prerequisites
- Node.js (v18+) and npm

### Run Development Server
```bash
npm install
npm run dev
```

Open `http://localhost:3000` in your mobile browser or PC.

> **Note on Camera Permissions**:
> Browsers require `localhost` or an `https://` connection to grant camera access. For testing on a physical phone over local Wi-Fi, use a secure tunnel (such as Cloudflare Tunnel or VS Code Port Forwarding with HTTPS).

### Production Build
```bash
npm run build
npm run preview
```
The output in `dist/` is a pure static web application that can be hosted on GitHub Pages, Vercel, Netlify, or any static file host.

---

## 3. Project Structure

```
Edge_Inference/
├── index.html                   # HTML entrypoint with ONNX Runtime Web
├── vite.config.js               # Vite configuration with COOP/COEP headers
├── package.json                 # React 18, onnxruntime-web, lucide-react
├── public/
│   └── manifest.json            # PWA manifest
├── src/
│   ├── main.jsx                 # React root mount
│   ├── App.jsx                  # Master state, camera coordinator, and loop
│   ├── index.css                # Mobile-first high-contrast dark theme
│   ├── config/
│   │   └── modelCatalog.js      # Built-in model catalog, classes, and CDN URLs
│   ├── core/
│   │   ├── cache.js             # IndexedDB model binary storage
│   │   ├── camera.js            # WebRTC camera with 1:1 square crop & letterboxing
│   │   ├── downloader.js        # Streaming CDN downloader & local file reader
│   │   ├── postprocess.js       # Anchor decoding & NMS filter
│   │   ├── distance.js          # Area-based distance heuristic fallback
│   │   ├── spatialFusion.js     # Median depth sampling & corridor collision analysis
│   │   ├── temporalTracker.js   # Multi-object tracking & EMA smoothing
│   │   ├── navigationState.js   # 3-sector path clearance & SpeechSynthesis TTS
│   │   └── renderer.js          # Bounding box & corridor overlay renderer
│   ├── engine/
│   │   ├── yoloModel.js         # ONNX Runtime Web YOLO inference session
│   │   ├── depthModel.js        # ONNX Runtime Web Depth model session & colormap
│   │   └── modelManager.js      # Async multi-model coordinator
│   └── components/
│       ├── Header.jsx           # App header with offline/cache status
│       ├── SetupScreen.jsx      # Model selector, CDN download, file picker & parameters
│       ├── LiveScreen.jsx       # 1:1 Viewport, HUD dashboard, colormap & controls
│       ├── NavigationBanner.jsx # Top path clearance & hazard alert banner
│       └── HudDashboard.jsx     # Live telemetry (FPS, Latency, Sectors)
├── docs/                        # Architecture documentation
└── README.md
```

---

## 4. How Offline Model Storage Works

1. In the **Setup Screen**, select your desired YOLO model and Depth model.
2. Tap **Get YOLO** / **Get Depth** to download once from CDN, or use **Load Custom .onnx from Device Disk** to load your own weights without internet.
3. The model binary is stored in IndexedDB (`ROD_Edge_Models_DB`).
4. Once cached, the button turns green (`✓ Ready in Storage`) and detection functions completely offline.
