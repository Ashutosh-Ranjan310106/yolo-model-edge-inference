/**
 * DepthModel: Encapsulates on-device Dense Depth Estimation.
 * Supports:
 * - Metric depth (YOLO-Depth)
 * - Relative depth (Depth Anything V2) mapped to approximate metric meters
 * - Central core median ROI extraction
 * - Colormap rendering overlay
 */

function getOrt() {
  if (typeof window !== "undefined" && window.ort) {
    return window.ort;
  }
  throw new Error("ONNX Runtime Web is not loaded. Check internet connection or CDN script.");
}

export class DepthModel {
  constructor() {
    this.session = null;
    this.modelId = null;
    this.resolution = 512;
    this.provider = "wasm";
    this.inputTensorName = "images";
    this.outputTensorName = "depth";
    this.inputShape = [1, 3, 512, 512];
    this.inputBuffer = null;
    this.isMetric = true;
    this.isLoaded = false;
    this.latencyMs = 0;

    this.mean = [0.485, 0.456, 0.406];
    this.std = [0.229, 0.224, 0.225];
    this.isImageNetNorm = false;

    this.latestDepthMap = null;
    this.depthWidth = 512;
    this.depthHeight = 512;
    this.minDepth = 0.5;
    this.maxDepth = 10.0;
    this.letterboxInfo = null;
  }

  async load(arrayBuffer, metadata = {}) {
    const ort = getOrt();

    this.modelId = metadata.id || "depth_model";
    this.resolution = metadata.resolution || 512;
    this.inputShape = [1, 3, this.resolution, this.resolution];
    this.inputBuffer = new Float32Array(1 * 3 * this.resolution * this.resolution);
    this.inputTensorName = metadata.inputTensorName || (this.modelId.includes("anything") ? "pixel_values" : "images");
    this.outputTensorName = metadata.outputTensorName || (this.modelId.includes("anything") ? "predicted_depth" : "depth");
    this.isMetric = metadata.isMetric !== undefined ? metadata.isMetric : !this.modelId.includes("anything");

    if (this.modelId.includes("anything")) {
      this.mean = [0.485, 0.456, 0.406];
      this.std = [0.229, 0.224, 0.225];
      this.isImageNetNorm = true;
    } else {
      this.mean = [0, 0, 0];
      this.std = [1, 1, 1];
      this.isImageNetNorm = false;
    }

    if (ort.env && ort.env.wasm) {
      ort.env.wasm.simd = true;
      ort.env.wasm.proxy = false;
      ort.env.wasm.numThreads = 1;
    }

    const sessionOptions = {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "disabled"
    };

    console.log(`[DepthModel] Initializing ${this.modelId} (${this.resolution}x${this.resolution}, metric=${this.isMetric}) with WASM SIMD...`);
    this.session = await ort.InferenceSession.create(arrayBuffer, sessionOptions);
    this.provider = "wasm";
    this.isLoaded = true;

    return {
      modelId: this.modelId,
      resolution: this.resolution,
      provider: this.provider,
      isMetric: this.isMetric
    };
  }

  preprocess(imageData) {
    const ort = getOrt();
    const { data, width, height } = imageData;
    const channelSize = width * height;
    const dst = this.inputBuffer;

    let srcIdx = 0;
    if (this.isImageNetNorm) {
      const [mR, mG, mB] = this.mean;
      const [sR, sG, sB] = this.std;
      for (let i = 0; i < channelSize; i++) {
        dst[i] = (data[srcIdx] / 255.0 - mR) / sR;
        dst[channelSize + i] = (data[srcIdx + 1] / 255.0 - mG) / sG;
        dst[2 * channelSize + i] = (data[srcIdx + 2] / 255.0 - mB) / sB;
        srcIdx += 4;
      }
    } else {
      for (let i = 0; i < channelSize; i++) {
        dst[i] = data[srcIdx] / 255.0;
        dst[channelSize + i] = data[srcIdx + 1] / 255.0;
        dst[2 * channelSize + i] = data[srcIdx + 2] / 255.0;
        srcIdx += 4;
      }
    }

    return new ort.Tensor("float32", dst, this.inputShape);
  }

  async predict(imageData, letterboxInfo = null) {
    if (!this.session || !this.isLoaded) return null;

    this.letterboxInfo = letterboxInfo;
    const tStart = performance.now();

    const tensor = this.preprocess(imageData);
    const inName = this.session.inputNames[0] || this.inputTensorName;
    const feeds = {};
    feeds[inName] = tensor;

    const results = await this.session.run(feeds);
    const tInf = performance.now();

    const outName = this.session.outputNames[0] || this.outputTensorName;
    const outputTensor = results[outName];

    const dims = outputTensor.dims;
    let h, w;
    if (dims.length === 3) {
      [, h, w] = dims;
    } else if (dims.length === 4) {
      [, , h, w] = dims;
    } else {
      h = Math.round(Math.sqrt(outputTensor.data.length));
      w = h;
    }

    const rawData = outputTensor.data;
    const totalPixels = h * w;
    const depthMap = new Float32Array(totalPixels);

    let minD = Infinity;
    let maxD = -Infinity;

    if (this.isMetric) {
      for (let i = 0; i < totalPixels; i++) {
        const val = Math.max(0.1, Math.min(rawData[i], 25.0));
        depthMap[i] = val;
        if (val < minD) minD = val;
        if (val > maxD) maxD = val;
      }
    } else {
      for (let i = 0; i < totalPixels; i++) {
        const val = rawData[i];
        if (val < minD) minD = val;
        if (val > maxD) maxD = val;
      }
      const range = maxD - minD || 1.0;
      for (let i = 0; i < totalPixels; i++) {
        const norm = (rawData[i] - minD) / range;
        depthMap[i] = 0.6 + (1.0 - norm) * 11.4;
      }
      minD = 0.6;
      maxD = 12.0;
    }

    const tEnd = performance.now();
    this.latencyMs = Math.round(tEnd - tStart);
    this.latestDepthMap = depthMap;
    this.depthWidth = w;
    this.depthHeight = h;
    this.minDepth = minD;
    this.maxDepth = maxD;

    return {
      depthMap,
      width: w,
      height: h,
      minDepth: minD,
      maxDepth: maxD,
      latencyMs: this.latencyMs,
      infLatencyMs: Math.round(tInf - tStart)
    };
  }

  getMedianDepthInROI(x1Norm, y1Norm, x2Norm, y2Norm) {
    if (!this.latestDepthMap) return null;

    if (this.letterboxInfo) {
      const { padX, padY, newW, newH, targetSize } = this.letterboxInfo;
      x1Norm = Math.max(0, Math.min(1, (padX + x1Norm * newW) / targetSize));
      x2Norm = Math.max(0, Math.min(1, (padX + x2Norm * newW) / targetSize));
      y1Norm = Math.max(0, Math.min(1, (padY + y1Norm * newH) / targetSize));
      y2Norm = Math.max(0, Math.min(1, (padY + y2Norm * newH) / targetSize));
    }

    const w = this.depthWidth;
    const h = this.depthHeight;

    const x1 = Math.max(0, Math.min(w - 1, Math.floor(x1Norm * w)));
    const y1 = Math.max(0, Math.min(h - 1, Math.floor(y1Norm * h)));
    const x2 = Math.max(0, Math.min(w - 1, Math.floor(x2Norm * w)));
    const y2 = Math.max(0, Math.min(h - 1, Math.floor(y2Norm * h)));

    if (x2 <= x1 || y2 <= y1) return null;

    // Sample central 50% core to avoid edge background bleeding
    const cx = (x1 + x2) / 2;
    const cy = (y1 + y2) / 2;
    const coreW = Math.max(2, (x2 - x1) * 0.5);
    const coreH = Math.max(2, (y2 - y1) * 0.5);

    const subX1 = Math.floor(cx - coreW / 2);
    const subX2 = Math.ceil(cx + coreW / 2);
    const subY1 = Math.floor(cy - coreH / 2);
    const subY2 = Math.ceil(cy + coreH / 2);

    const values = [];
    for (let y = subY1; y <= subY2; y++) {
      const rowOffset = y * w;
      for (let x = subX1; x <= subX2; x++) {
        const val = this.latestDepthMap[rowOffset + x];
        if (val > 0.05 && val < 30.0) {
          values.push(val);
        }
      }
    }

    if (values.length === 0) return null;
    values.sort((a, b) => a - b);
    return values[Math.floor(values.length / 2)];
  }

  renderColorMap(canvas, alpha = 0.65) {
    if (!this.latestDepthMap || !canvas) return;

    const w = this.depthWidth;
    const h = this.depthHeight;

    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }

    const ctx = canvas.getContext("2d");
    const imgData = ctx.createImageData(w, h);
    const data = imgData.data;

    const minD = this.minDepth;
    const maxD = Math.min(this.maxDepth, 10.0);
    const range = maxD - minD || 1.0;

    for (let i = 0; i < w * h; i++) {
      const d = Math.max(minD, Math.min(maxD, this.latestDepthMap[i]));
      const t = 1.0 - (d - minD) / range; // 1.0 = close (warm), 0.0 = far (cool)
      const pixelIdx = i * 4;

      // Turbo/Magma approximate gradient
      let r, g, b;
      if (t < 0.33) {
        const subT = t / 0.33;
        r = Math.round(30 + subT * 40);
        g = Math.round(50 + subT * 100);
        b = Math.round(180 + subT * 70);
      } else if (t < 0.66) {
        const subT = (t - 0.33) / 0.33;
        r = Math.round(70 + subT * 180);
        g = Math.round(150 + subT * 70);
        b = Math.round(250 - subT * 200);
      } else {
        const subT = (t - 0.66) / 0.34;
        r = 250;
        g = Math.round(220 - subT * 180);
        b = Math.round(50 - subT * 40);
      }

      data[pixelIdx] = r;
      data[pixelIdx + 1] = g;
      data[pixelIdx + 2] = b;
      data[pixelIdx + 3] = Math.round(255 * alpha);
    }

    ctx.putImageData(imgData, 0, 0);
  }

  dispose() {
    this.session = null;
    this.isLoaded = false;
    this.inputBuffer = null;
    this.latestDepthMap = null;
  }
}
