# ROD Edge Inference — Android Native Kotlin Application

A high-performance Android native application using **CameraX**, **Jetpack Compose**, and **ONNX Runtime Android** for low-latency on-device obstacle detection.

---

## Technical Stack

- **Language**: Kotlin
- **Camera API**: CameraX `ImageAnalysis` with `STRATEGY_KEEP_ONLY_LATEST`
- **Inference Runtime**: Microsoft ONNX Runtime Android (NNAPI hardware acceleration with CPU fallback)
- **UI Framework**: Jetpack Compose with custom hardware-accelerated Canvas overlay
- **Local Cache**: Internal app storage (`context.filesDir/models/`) with SHA-256 verification
- **Network**: OkHttp streaming client with progress tracking

---

## How to Build in Android Studio

1. Open Android Studio.
2. Select **Open** and navigate to `ROD_Edge_Inference/mobile`.
3. Let Gradle sync dependencies (`onnxruntime-android`, CameraX, Compose).
4. Connect an Android phone via USB with USB Debugging enabled.
5. In `MainActivity.kt`, verify or adjust the default LAN server URL to match your host machine's Wi-Fi IP address.
6. Click **Run 'app'**.
