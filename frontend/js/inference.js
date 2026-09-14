/**
 * On-Device Inference Engine powered by ONNX Runtime Web.
 * Executes YOLO26 models entirely inside the client's browser using WebGPU or WebAssembly SIMD.
 */
import { YOLOVisualPostProcessor } from "./postprocess.js";
import { DistanceEstimator } from "./distance.js";

export class EdgeInferenceEngine {
  constructor() {
    this.session = null;
    this.modelId = null;
    this.resolution = 480;
    this.executionProvider = "wasm";
    this.postProcessor = new YOLOVisualPostProcessor([]);
    this.distanceEstimator = new DistanceEstimator();

    // Reusable Float32Array buffer to eliminate GC pressure during live stream
    this.inputFloatBuffer = null;
    this.inputTensorShape = [1, 3, 480, 480];

    // FPS & Latency tracking
    this.lastFrameTime = performance.now();
    this.fps = 0;
    this.fpsHistory = [];
    this.latency = {
      preprocess: 0,
      inference: 0,
      postprocess: 0,
      total: 0
    };
  }

  /**
   * Load model from an ArrayBuffer
   */
  async loadModel(arrayBuffer, modelMeta = {}) {
    if (!window.ort) {
      throw new Error("ONNX Runtime Web (ort) is not loaded in the window.");
    }

    this.modelId = modelMeta.id || "yolo_model";
    this.resolution = modelMeta.resolution || 480;
    this.inputTensorShape = [1, 3, this.resolution, this.resolution];
    this.inputFloatBuffer = new Float32Array(1 * 3 * this.resolution * this.resolution);
    
    if (modelMeta.class_names) {
      this.postProcessor.setClasses(modelMeta.class_names);
    }

    // Try WebGPU first, then fall back to WASM
    const sessionOptions = {
      executionProviders: ["webgpu", "wasm"],
      graphOptimizationLevel: "all"
    };

    console.log(`[Engine] Initializing ONNX Runtime session for ${this.modelId} (${this.resolution}x${this.resolution})...`);
    
    try {
      this.session = await ort.InferenceSession.create(arrayBuffer, sessionOptions);
      this.executionProvider = ("gpu" in navigator) ? "webgpu" : "wasm";
      console.log(`[Engine] Successfully loaded with ${this.executionProvider.toUpperCase()} provider.`);
    } catch (gpuError) {
      console.warn("[Engine] WebGPU unavailable or failed, falling back to WASM:", gpuError);
      sessionOptions.executionProviders = ["wasm"];
      this.session = await ort.InferenceSession.create(arrayBuffer, sessionOptions);
      this.executionProvider = "wasm";
      console.log("[Engine] Successfully loaded with WASM SIMD provider.");
    }

    return {
      provider: this.executionProvider,
      inputNames: this.session.inputNames,
      outputNames: this.session.outputNames
    };
  }

  /**
   * Preprocess ImageData to planar NCHW float32 [1, 3, H, W] normalized to [0, 1]
   */
  preprocess(imageData) {
    const { data, width, height } = imageData;
    const channelSize = width * height;
    const dst = this.inputFloatBuffer;

    // Separate R, G, B channels and normalize 0..255 to 0..1
    let srcIdx = 0;
    for (let i = 0; i < channelSize; i++) {
      dst[i] = data[srcIdx] / 255.0;                   // R
      dst[channelSize + i] = data[srcIdx + 1] / 255.0; // G
      dst[2 * channelSize + i] = data[srcIdx + 2] / 255.0; // B
      srcIdx += 4;
    }

    return new ort.Tensor("float32", dst, this.inputTensorShape);
  }

  /**
   * Execute single inference cycle on pre-scaled frame
   */
  async runFrame(imageData, frameWidth, frameHeight, confThreshold = 0.25) {
    if (!this.session) {
      return { detections: [], latency: this.latency, fps: 0 };
    }

    const tStart = performance.now();

    // 1. Preprocess
    const tensor = this.preprocess(imageData);
    const tPre = performance.now();

    // 2. Inference
    const feeds = {};
    feeds[this.session.inputNames[0]] = tensor;
    const results = await this.session.run(feeds);
    const tInf = performance.now();

    // 3. Postprocess
    const outputTensor = results[this.session.outputNames[0]];
    const detections = this.postProcessor.decode(
      outputTensor.data,
      outputTensor.dims,
      this.resolution,
      frameWidth,
      frameHeight,
      confThreshold
    );

    // 4. Attach distance estimations
    for (const det of detections) {
      det.distance = this.distanceEstimator.estimate(det.box, frameWidth, frameHeight);
    }

    const tPost = performance.now();

    // Latency metrics
    this.latency = {
      preprocess: Math.round(tPre - tStart),
      inference: Math.round(tInf - tPre),
      postprocess: Math.round(tPost - tInf),
      total: Math.round(tPost - tStart)
    };

    // Calculate real measured FPS based on actual completion intervals
    const now = performance.now();
    const deltaMs = now - this.lastFrameTime;
    this.lastFrameTime = now;
    if (deltaMs > 0) {
      const instantFps = 1000 / deltaMs;
      this.fpsHistory.push(instantFps);
      if (this.fpsHistory.length > 10) this.fpsHistory.shift();
      this.fps = Number((this.fpsHistory.reduce((a, b) => a + b, 0) / this.fpsHistory.length).toFixed(1));
    }

    return {
      detections,
      latency: this.latency,
      fps: this.fps,
      provider: this.executionProvider
    };
  }

  isReady() {
    return !!this.session;
  }
}
