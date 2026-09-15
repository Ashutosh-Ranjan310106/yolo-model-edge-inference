/**
 * Camera Pipeline Manager for Mobile Edge Inference.
 * Implements non-blocking 'Latest Frame' strategy:
 * - 1:1 center-square crop for YOLO
 * - Neutral gray letterboxing for depth models
 */

export class MobileCameraManager {
  constructor(videoElement) {
    this.video = videoElement;
    this.stream = null;
    this.isPlaying = false;
    this.frameCanvas = document.createElement("canvas");
    this.frameCtx = this.frameCanvas.getContext("2d", { willReadFrequently: true });
  }

  async start(facingMode = "environment") {
    if (this.stream) {
      this.stop();
    }

    const constraints = {
      audio: false,
      video: {
        facingMode: { ideal: facingMode },
        width: { ideal: 1280 },
        height: { ideal: 720 }
      }
    };

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error(
        "Camera API is not supported or blocked by browser security.\n" +
        "Ensure the app is running on localhost or over HTTPS."
      );
    }

    try {
      this.stream = await navigator.mediaDevices.getUserMedia(constraints);
      if (this.video) {
        this.video.srcObject = this.stream;
        await new Promise((resolve) => {
          this.video.onloadedmetadata = () => {
            this.video.play();
            this.isPlaying = true;
            this.frameCanvas.width = this.video.videoWidth || 640;
            this.frameCanvas.height = this.video.videoHeight || 480;
            resolve();
          };
        });
      }
      return true;
    } catch (err) {
      console.error("[Camera] Access failed:", err);
      throw err;
    }
  }

  stop() {
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }
    this.isPlaying = false;
    if (this.video) {
      this.video.srcObject = null;
    }
  }

  /**
   * Capture center 1:1 square crop of camera frame into ImageData
   */
  captureSquareFrame(size) {
    if (!this.isPlaying || !this.video || this.video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      return null;
    }

    const vW = this.video.videoWidth;
    const vH = this.video.videoHeight;
    if (!vW || !vH) return null;

    const cropSize = Math.min(vW, vH);
    const cropX = Math.round((vW - cropSize) / 2);
    const cropY = Math.round((vH - cropSize) / 2);

    if (this.frameCanvas.width !== size || this.frameCanvas.height !== size) {
      this.frameCanvas.width = size;
      this.frameCanvas.height = size;
    }

    this.frameCtx.drawImage(
      this.video,
      cropX, cropY, cropSize, cropSize,
      0, 0, size, size
    );

    return this.frameCtx.getImageData(0, 0, size, size);
  }

  /**
   * Capture letterbox frame preserving native aspect ratio with neutral gray padding
   */
  captureLetterboxFrame(size) {
    if (!this.isPlaying || !this.video || this.video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      return null;
    }

    const vW = this.video.videoWidth;
    const vH = this.video.videoHeight;
    if (!vW || !vH) return null;

    if (this.frameCanvas.width !== size || this.frameCanvas.height !== size) {
      this.frameCanvas.width = size;
      this.frameCanvas.height = size;
    }

    const scale = Math.min(size / vW, size / vH);
    const newW = Math.round(vW * scale);
    const newH = Math.round(vH * scale);
    const padX = Math.round((size - newW) / 2);
    const padY = Math.round((size - newH) / 2);

    // Standard YOLO neutral gray background (114, 114, 114)
    this.frameCtx.fillStyle = "rgb(114, 114, 114)";
    this.frameCtx.fillRect(0, 0, size, size);

    this.frameCtx.drawImage(this.video, 0, 0, vW, vH, padX, padY, newW, newH);

    const imageData = this.frameCtx.getImageData(0, 0, size, size);

    return {
      imageData,
      letterboxInfo: {
        scale,
        padX,
        padY,
        newW,
        newH,
        targetSize: size,
        origWidth: vW,
        origHeight: vH
      }
    };
  }

  getDimensions() {
    return {
      width: this.video?.videoWidth || 640,
      height: this.video?.videoHeight || 480
    };
  }
}
