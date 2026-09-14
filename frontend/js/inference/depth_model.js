/**
 * DepthModel: Encapsulates on-device Dense Depth Estimation using Depth Anything V2 / YOLO Depth.
 * Computes per-pixel depth maps in meters or normalized proximity via ONNX Runtime Web (WebGPU / WASM).
 */

export class DepthModel {
  constructor() {
    this.session = null;
    this.modelId = null;
    this.resolution = 518;
    this.provider = "wasm";
    this.inputTensorName = "pixel_values";
    this.outputTensorName = "predicted_depth";
    this.inputShape = [1, 3, 518, 518];
    this.inputBuffer = null;
    this.isMetric = true;
    this.isLoaded = false;
    this.latencyMs = 0;

    // Normalization parameters
    this.mean = [0.485, 0.456, 0.406];
    this.std = [0.229, 0.224, 0.225];
    this.isImageNetNorm = true;

    // Latest depth cache
    this.latestDepthMap = null; // Float32Array
    this.depthWidth = 518;
    this.depthHeight = 518;
    this.minDepth = 0;
    this.maxDepth = 10;
  }

  async load(arrayBuffer, metadata = {}) {
    if (!window.ort) {
      throw new Error("ONNX Runtime Web (ort) not loaded.");
    }

    this.modelId = metadata.id || "depth_model";
    this.resolution = metadata.resolution || 518;
    this.inputShape = [1, 3, this.resolution, this.resolution];
    this.inputBuffer = new Float32Array(1 * 3 * this.resolution * this.resolution);
    this.inputTensorName = metadata.input_tensor_name || "pixel_values";
    this.outputTensorName = metadata.output_tensor_name || "predicted_depth";

    if (metadata.distance_heuristic && metadata.distance_heuristic.is_metric !== undefined) {
      this.isMetric = metadata.distance_heuristic.is_metric;
    } else {
      this.isMetric = this.modelId.includes("metric") || this.modelId.includes("depth");
    }

    if (metadata.mean && metadata.std) {
      this.mean = metadata.mean;
      this.std = metadata.std;
      this.isImageNetNorm = (this.std[0] < 1.0);
    } else if (this.modelId.includes("yolo")) {
      this.mean = [0, 0, 0];
      this.std = [255, 255, 255];
      this.isImageNetNorm = false;
    } else {
      this.mean = [0.485, 0.456, 0.406];
      this.std = [0.229, 0.224, 0.225];
      this.isImageNetNorm = true;
    }

    // Configure WASM runtime performance (proxy=false and numThreads=1 avoids cross-origin Worker SecurityError on CDN scripts)
    if (window.ort && window.ort.env && window.ort.env.wasm) {
      window.ort.env.wasm.simd = true;
      window.ort.env.wasm.proxy = false;
      window.ort.env.wasm.numThreads = 1;
    }

    const sessionOptions = {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "disabled"
    };

    console.log(`[DepthModel] Fast-loading ${this.modelId} (${this.resolution}x${this.resolution}, metric=${this.isMetric}) with WASM SIMD provider...`);

    this.session = await ort.InferenceSession.create(arrayBuffer, sessionOptions);
    this.provider = "wasm";
    console.log(`[DepthModel] Session created successfully with WASM provider.`);

    this.isLoaded = true;
    return {
      modelId: this.modelId,
      resolution: this.resolution,
      provider: this.provider,
      isMetric: this.isMetric
    };
  }

  preprocess(imageData) {
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

  async predict(imageData) {
    if (!this.session || !this.isLoaded) {
      return null;
    }

    const tStart = performance.now();

    const tensor = this.preprocess(imageData);
    const inName = this.session.inputNames[0] || this.inputTensorName;
    const feeds = {};
    feeds[inName] = tensor;

    const results = await this.session.run(feeds);
    const tInf = performance.now();

    const outName = this.session.outputNames[0] || this.outputTensorName;
    const outputTensor = results[outName];

    // Determine dimensions
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
      // Metric models: raw values are meters directly
      for (let i = 0; i < totalPixels; i++) {
        const val = rawData[i];
        const m = Math.max(0.1, Math.min(val, 25.0)); // Clamp 0.1m - 25m
        depthMap[i] = m;
        if (m < minD) minD = m;
        if (m > maxD) maxD = m;
      }
    } else {
      // Relative models: higher values mean closer or inverted depending on model
      // Find min and max for normalization
      for (let i = 0; i < totalPixels; i++) {
        const val = rawData[i];
        if (val < minD) minD = val;
        if (val > maxD) maxD = val;
      }
      const range = (maxD - minD) || 1.0;
      // Map to approximate metric meters (0.5m to 12.0m)
      for (let i = 0; i < totalPixels; i++) {
        const norm = (rawData[i] - minD) / range; // 0..1 (closer is higher in Depth Anything)
        // Inverse mapping: norm 1.0 -> 0.6m, norm 0.0 -> 12m
        const approxMeters = 0.6 + (1.0 - norm) * 11.4;
        depthMap[i] = approxMeters;
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
      isMetric: this.isMetric,
      latencyMs: this.latencyMs,
      infLatencyMs: Math.round(tInf - tStart)
    };
  }

  /**
   * Sample median depth in an ROI [x1, y1, x2, y2] normalized (0..1)
   */
  getMedianDepthInROI(x1Norm, y1Norm, x2Norm, y2Norm) {
    if (!this.latestDepthMap) return null;

    const w = this.depthWidth;
    const h = this.depthHeight;

    const x1 = Math.max(0, Math.min(w - 1, Math.floor(x1Norm * w)));
    const y1 = Math.max(0, Math.min(h - 1, Math.floor(y1Norm * h)));
    const x2 = Math.max(0, Math.min(w - 1, Math.floor(x2Norm * w)));
    const y2 = Math.max(0, Math.min(h - 1, Math.floor(y2Norm * h)));

    if (x2 <= x1 || y2 <= y1) return null;

    // Use central core (inner 50% box) to avoid background bleed
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
        values.push(this.latestDepthMap[rowOffset + x]);
      }
    }

    if (values.length === 0) return null;
    values.sort((a, b) => a - b);

    // Median
    const mid = Math.floor(values.length / 2);
    const median = values.length % 2 === 0
      ? (values[mid - 1] + values[mid]) / 2
      : values[mid];

    return Math.round(median * 10) / 10;
  }

  /**
   * Render colorized depth map overlay onto an HTML5 canvas.
   * Turbo / Inferno color ramp: Close (red/orange) -> Mid (yellow/green) -> Far (blue/purple).
   */
  renderColorMap(targetCanvas, alpha = 0.6) {
    if (!this.latestDepthMap || !targetCanvas) return;

    const w = this.depthWidth;
    const h = this.depthHeight;

    if (targetCanvas.width !== w || targetCanvas.height !== h) {
      targetCanvas.width = w;
      targetCanvas.height = h;
    }

    const ctx = targetCanvas.getContext("2d");
    const imgData = ctx.createImageData(w, h);
    const buf = imgData.data;

    const minD = this.minDepth;
    const maxD = Math.min(this.maxDepth, 15.0);
    const range = (maxD - minD) || 1.0;
    const map = this.latestDepthMap;

    const alphaInt = Math.round(alpha * 255);

    for (let i = 0; i < map.length; i++) {
      const d = map[i];
      // Normalize: 0 = close, 1 = far
      const norm = Math.max(0, Math.min(1, (d - minD) / range));
      
      // Color ramp: close (hot) -> far (cool)
      // Turbo-like gradient
      const r = Math.round(Math.sin((1.0 - norm) * Math.PI * 0.8) * 255);
      const g = Math.round(Math.sin((1.0 - norm) * Math.PI * 1.5) * 255);
      const b = Math.round(Math.sin(norm * Math.PI * 0.8) * 255);

      const idx = i * 4;
      buf[idx + 0] = Math.max(0, Math.min(255, r));
      buf[idx + 1] = Math.max(0, Math.min(255, g));
      buf[idx + 2] = Math.max(0, Math.min(255, b));
      buf[idx + 3] = alphaInt;
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
