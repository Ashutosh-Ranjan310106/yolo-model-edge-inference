/**
 * Camera Pipeline Manager for Mobile Edge Inference.
 * Implements a non-blocking 'Latest Frame' strategy to prevent buffering latency.
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
      const isHttp = window.location.protocol === "http:" && window.location.hostname !== "localhost" && window.location.hostname !== "127.0.0.1";
      if (isHttp) {
        throw new Error(
          "Camera access is blocked by mobile browser security on plain HTTP addresses.\n\n" +
          "👉 Solution: Open the app using your HTTPS DevTunnel URL:\n" +
          "https://gpg3lkpj-8000.inc1.devtunnels.ms/client/\n\n" +
          "Mobile browsers only grant camera permissions over HTTPS or localhost."
        );
      }
      throw new Error("Camera API is not supported on this browser.");
    }

    try {
      this.stream = await navigator.mediaDevices.getUserMedia(constraints);
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
      return true;
    } catch (err) {
      console.error("Camera access failed:", err);
      throw err;
    }
  }

  stop() {
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
    this.isPlaying = false;
    if (this.video) {
      this.video.srcObject = null;
    }
  }

  /**
   * Capture the center 1:1 square crop of the camera frame into offscreen canvas
   * @param {number} size - Target square resolution (e.g. 320, 480, 640)
   * @returns {ImageData|null}
   */
  captureSquareFrame(size) {
    if (!this.isPlaying || this.video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      return null;
    }

    const vW = this.video.videoWidth;
    const vH = this.video.videoHeight;
    if (!vW || !vH) return null;

    // Automatic 1:1 center-crop calculation
    const cropSize = Math.min(vW, vH);
    const cropX = Math.round((vW - cropSize) / 2);
    const cropY = Math.round((vH - cropSize) / 2);

    if (this.frameCanvas.width !== size || this.frameCanvas.height !== size) {
      this.frameCanvas.width = size;
      this.frameCanvas.height = size;
    }

    // Draw center square crop scaled to target size
    this.frameCtx.drawImage(
      this.video,
      cropX, cropY, cropSize, cropSize,
      0, 0, size, size
    );

    return this.frameCtx.getImageData(0, 0, size, size);
  }

  /**
   * Capture the center 1:1 square crop as a compressed JPEG data URL
   * @param {number} size - Target square resolution (e.g. 320, 480, 640)
   * @param {number} quality - JPEG compression quality (0.0 to 1.0)
   * @returns {string|null}
   */
  captureSquareJpeg(size, quality = 0.7) {
    if (!this.isPlaying || this.video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
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

    return this.frameCanvas.toDataURL("image/jpeg", quality);
  }

  /**
   * Capture frame at target resolution, applying 1:1 center-crop if target is square
   */
  captureResizedFrame(targetWidth, targetHeight) {
    if (targetWidth === targetHeight) {
      return this.captureSquareFrame(targetWidth);
    }

    if (!this.isPlaying || this.video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      return null;
    }

    if (this.frameCanvas.width !== targetWidth || this.frameCanvas.height !== targetHeight) {
      this.frameCanvas.width = targetWidth;
      this.frameCanvas.height = targetHeight;
    }

    this.frameCtx.drawImage(this.video, 0, 0, targetWidth, targetHeight);
    return this.frameCtx.getImageData(0, 0, targetWidth, targetHeight);
  }

  getDimensions() {
    return {
      width: this.video.videoWidth || 640,
      height: this.video.videoHeight || 480
    };
  }
}
