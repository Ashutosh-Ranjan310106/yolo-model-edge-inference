/**
 * ModelManager: Central orchestrator managing YoloModel and DepthModel independently.
 * Coordinates asynchronous execution, rate-limiting, and latest-frame buffering.
 */

import { YoloModel } from "./yoloModel.js";
import { DepthModel } from "./depthModel.js";

export class ModelManager {
  constructor() {
    this.yolo = new YoloModel();
    this.depth = new DepthModel();

    this.yoloTargetFps = 15;
    this.depthTargetFps = 6;

    this.stats = {
      yoloFps: 0,
      yoloLatencyMs: 0,
      depthFps: 0,
      depthLatencyMs: 0,
      depthResolution: 512,
      depthDroppedFrames: 0,
      yoloProvider: "wasm",
      depthProvider: "wasm",
      lastError: null
    };

    this.lastYoloRunTime = 0;
    this.lastDepthRunTime = 0;
    this.isYoloRunning = false;
    this.isDepthRunning = false;

    this.yoloFpsHistory = [];
    this.depthFpsHistory = [];

    this.latestDetections = [];
    this.latestDepthResult = null;
  }

  async loadYoloModel(arrayBuffer, metadata) {
    const res = await this.yolo.load(arrayBuffer, metadata);
    this.stats.yoloProvider = res.provider;
    this.stats.lastError = null;
    return res;
  }

  async loadDepthModel(arrayBuffer, metadata) {
    const res = await this.depth.load(arrayBuffer, metadata);
    this.stats.depthProvider = res.provider;
    this.stats.depthResolution = res.resolution || metadata.resolution || 512;
    this.stats.depthDroppedFrames = 0;
    this.stats.lastError = null;
    return res;
  }

  hasYolo() {
    return this.yolo.isLoaded;
  }

  hasDepth() {
    return this.depth.isLoaded;
  }

  async maybeRunYolo(imageData, frameWidth, frameHeight, confThreshold) {
    if (!this.yolo.isLoaded || this.isYoloRunning) {
      return this.latestDetections;
    }

    const now = performance.now();
    const interval = 1000 / this.yoloTargetFps;
    if (now - this.lastYoloRunTime < interval) {
      return this.latestDetections;
    }

    this.isYoloRunning = true;
    try {
      const res = await this.yolo.predict(imageData, frameWidth, frameHeight, confThreshold);
      this.latestDetections = res.detections;
      this.stats.yoloLatencyMs = res.latencyMs;
      this.stats.lastError = null;

      const delta = now - this.lastYoloRunTime;
      if (delta > 0 && this.lastYoloRunTime > 0) {
        this.yoloFpsHistory.push(1000 / delta);
        if (this.yoloFpsHistory.length > 10) this.yoloFpsHistory.shift();
        this.stats.yoloFps = Number((this.yoloFpsHistory.reduce((a, b) => a + b, 0) / this.yoloFpsHistory.length).toFixed(1));
      }
      this.lastYoloRunTime = now;
    } catch (err) {
      console.warn("[ModelManager] YOLO inference error:", err);
      this.stats.lastError = err.message || "YOLO inference error";
    } finally {
      this.isYoloRunning = false;
    }

    return this.latestDetections;
  }

  async maybeRunDepth(imageData, letterboxInfo = null) {
    if (!this.depth.isLoaded) {
      return this.latestDepthResult;
    }

    if (this.isDepthRunning) {
      this.stats.depthDroppedFrames++;
      return this.latestDepthResult;
    }

    const now = performance.now();
    const interval = 1000 / this.depthTargetFps;
    if (now - this.lastDepthRunTime < interval) {
      return this.latestDepthResult;
    }

    this.isDepthRunning = true;
    try {
      const res = await this.depth.predict(imageData, letterboxInfo);
      if (res) {
        this.latestDepthResult = res;
        this.stats.depthLatencyMs = res.latencyMs;
        this.stats.lastError = null;

        const delta = now - this.lastDepthRunTime;
        if (delta > 0 && this.lastDepthRunTime > 0) {
          this.depthFpsHistory.push(1000 / delta);
          if (this.depthFpsHistory.length > 10) this.depthFpsHistory.shift();
          this.stats.depthFps = Number((this.depthFpsHistory.reduce((a, b) => a + b, 0) / this.depthFpsHistory.length).toFixed(1));
        }
        this.lastDepthRunTime = now;
      }
    } catch (err) {
      console.warn("[ModelManager] Depth inference error:", err);
      this.stats.lastError = err.message || "Depth inference error";
    } finally {
      this.isDepthRunning = false;
    }

    return this.latestDepthResult;
  }

  dispose() {
    this.yolo.dispose();
    this.depth.dispose();
    this.latestDetections = [];
    this.latestDepthResult = null;
  }
}
