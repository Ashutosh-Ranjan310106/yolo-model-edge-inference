/**
 * YoloModel: Encapsulates YOLO object detection inference via ONNX Runtime Web.
 * Runs on WebGPU or WASM SIMD, completely on-device.
 */
import { YOLOVisualPostProcessor } from "../postprocess.js";

export class YoloModel {
  constructor() {
    this.session = null;
    this.modelId = null;
    this.resolution = 480;
    this.provider = "wasm";
    this.postProcessor = new YOLOVisualPostProcessor([]);
    this.inputShape = [1, 3, 480, 480];
    this.inputBuffer = null;
    this.inputTensorName = "images";
    this.outputTensorName = "output0";
    this.isLoaded = false;
    this.latencyMs = 0;
  }

  async load(arrayBuffer, metadata = {}) {
    if (!window.ort) {
      throw new Error("ONNX Runtime Web (ort) not loaded.");
    }

    this.modelId = metadata.id || "yolo_model";
    this.resolution = metadata.resolution || 480;
    this.inputShape = [1, 3, this.resolution, this.resolution];
    this.inputBuffer = new Float32Array(1 * 3 * this.resolution * this.resolution);
    this.inputTensorName = metadata.input_tensor_name || "images";
    this.outputTensorName = metadata.output_tensor_name || "output0";

    if (metadata.class_names && metadata.class_names.length > 0) {
      this.postProcessor.setClasses(metadata.class_names);
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

    console.log(`[YoloModel] Fast-loading ${this.modelId} (${this.resolution}x${this.resolution}) with WASM SIMD provider...`);

    this.session = await ort.InferenceSession.create(arrayBuffer, sessionOptions);
    this.provider = "wasm";
    console.log(`[YoloModel] Session created successfully with WASM provider.`);

    this.arrayBuffer = arrayBuffer;
    this.isLoaded = true;
    return {
      modelId: this.modelId,
      resolution: this.resolution,
      provider: this.provider
    };
  }

  preprocess(imageData) {
    const { data, width, height } = imageData;
    const channelSize = width * height;
    const dst = this.inputBuffer;

    let srcIdx = 0;
    for (let i = 0; i < channelSize; i++) {
      dst[i] = data[srcIdx] / 255.0;                   // R
      dst[channelSize + i] = data[srcIdx + 1] / 255.0; // G
      dst[2 * channelSize + i] = data[srcIdx + 2] / 255.0; // B
      srcIdx += 4;
    }

    return new ort.Tensor("float32", dst, this.inputShape);
  }

  async predict(imageData, frameWidth, frameHeight, confThreshold = 0.25) {
    if (!this.session || !this.isLoaded) {
      return { detections: [], latencyMs: 0 };
    }

    const tStart = performance.now();

    const tensor = this.preprocess(imageData);
    const inName = this.session.inputNames[0] || this.inputTensorName;
    const feeds = {};
    feeds[inName] = tensor;

    let results;
    try {
      results = await this.session.run(feeds);
    } catch (err) {
      if (this.provider === "webgpu" && this.arrayBuffer) {
        console.warn("[YoloModel] WebGPU execution failed at runtime. Falling back to WASM SIMD provider:", err);
        const sessionOptions = {
          executionProviders: ["wasm"],
          graphOptimizationLevel: "basic"
        };
        this.session = await ort.InferenceSession.create(this.arrayBuffer, sessionOptions);
        this.provider = "wasm";
        results = await this.session.run(feeds);
      } else {
        throw err;
      }
    }
    const tInf = performance.now();

    const outName = this.session.outputNames[0] || this.outputTensorName;
    const outputTensor = results[outName];

    const detections = this.postProcessor.decode(
      outputTensor.data,
      outputTensor.dims,
      this.resolution,
      frameWidth,
      frameHeight,
      confThreshold
    );

    const tEnd = performance.now();
    this.latencyMs = Math.round(tEnd - tStart);

    return {
      detections,
      latencyMs: this.latencyMs,
      infLatencyMs: Math.round(tInf - tStart)
    };
  }

  dispose() {
    this.session = null;
    this.isLoaded = false;
    this.inputBuffer = null;
    this.arrayBuffer = null;
  }
}
