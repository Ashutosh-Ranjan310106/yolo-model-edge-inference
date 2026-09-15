# Architecture Specification — Standalone ROD Edge Inference System

## 1. System Topology (Standalone Edge React Client)

```
+-------------------------------------------------------------------------+
|                  PUBLIC CDN (Hugging Face / GitHub)                     |
|                                   |                                     |
|                                   | (One-Time Model Download / Cache)   |
|                                   v                                     |
|                     OR LOCAL .ONNX FILE PICKER                          |
+-----------------------------------|-------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                    CLIENT DEVICE (PHONE / BROWSER EDGE)                 |
|                                                                         |
|   +-----------------------------------------------------------------+   |
|   |                       INDEXEDDB LOCAL CACHE                     |   |
|   |                  (ROD_Edge_Models_DB Store)                     |   |
|   +-----------------------------------------------------------------+   |
|                                  |                                      |
|                                  v                                      |
|    +-------------+      +-------------------+      +----------------+   |
|    |  WebRTC Cam | ---> | 1:1 Center-Square | ---> | YOLO Inference |   |
|    |  Feed       |      | & Letterbox Pad   |      | (ONNX Web SIMD)|   |
|    +-------------+      +-------------------+      +----------------+   |
|                                                            |            |
|                                                            v            |
|    +-------------+      +-------------------+      +----------------+   |
|    | Audio Voice | <--- | Spatial Corridor  | <--- | Postprocessing |   |
|    | & HUD View  |      | & Depth Fusion    |      | (Decode & NMS) |   |
|    +-------------+      +-------------------+      +----------------+   |
|                                                                         |
+-------------------------------------------------------------------------+
```

## 2. Zero Backend Guarantee
- **No Backend Server**: Zero Python, zero FastAPI, zero server-side GPUs.
- **100% On-Device Processing**: Camera frames and inference remain strictly in browser RAM on the client device.
- **Zero Data Leakage**: No images or frames ever leave the phone.
- **Offline Capable**: Once model weights are saved in IndexedDB, the entire application works offline in Airplane Mode.

## 3. Latest-Frame Pipeline Strategy
```
Camera Frame 1  ---> [ Inference Active ] ---> Render HUD & Boxes
Camera Frame 2  ---> [ Busy! DISCARD   ]
Camera Frame 3  ---> [ Busy! DISCARD   ]
Camera Frame 4  ---> [ Busy! DISCARD   ]
Inference Done!
Camera Frame 5  ---> [ Newest Frame     ] ---> [ Inference Active ]
```
This ensures the camera preview remains smooth at 30–60 FPS with zero queued latency backlog.

## 4. Multi-Sensor Perception Fusion
- **YOLO Detection (~15 FPS)**: 2D bounding boxes and object classification (27/25 obstacle classes).
- **Dense Depth Estimation (~6 FPS)**: Metric depth map calculation or relative depth normalization.
- **Spatial Fusion**: Median depth extraction in bounding box ROIs, 7-zone direction classification, 5 distance tiers, 3-sector path clearance, and walking corridor trapezoid hazard overlap.
- **Temporal Tracking**: Exponential moving average smoothing and track ID persistence over dropped frames.
- **Speech Guidance**: Web Speech Synthesis API alerts visually impaired users of upcoming hazards in real-time.
