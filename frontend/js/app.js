/**
 * ROD Edge Visual Navigation System - Master Application.
 * Core Pipeline:
 * Phone Camera -> Frame Preproc -> Local YOLO ONNX -> Local Depth ONNX
 * -> Spatial Fusion -> Temporal Tracker -> Navigation State -> Live UI Viewport
 */

import { ModelRegistryAPI } from "./api.js?v=4.8";
import { modelCache } from "./cache.js?v=4.8";
import { MobileCameraManager } from "./camera.js?v=4.8";
import { ModelManager } from "./inference/model_manager.js?v=4.8";
import { SpatialDepthFusion } from "./fusion/spatial_fusion.js?v=4.8";
import { TemporalTracker } from "./tracking/temporal_tracker.js?v=4.8";
import { NavigationState } from "./navigation/navigation_state.js?v=4.8";
import { pcClient } from "./pc_client.js?v=4.8";

class App {
  constructor() {
    this.api = new ModelRegistryAPI();
    this.modelManager = new ModelManager();
    this.spatialFusion = new SpatialDepthFusion();
    this.temporalTracker = new TemporalTracker();
    this.navigationState = new NavigationState();
    this.camera = null;

    // Inference mode: "edge" (phone on-device) or "pc" (stream to PC)
    this.inferenceMode = "edge";

    // PC Mode settings
    this.pcRecordEnabled = true;
    this.pcRecordFreq = 5;
    this.pcOnlyDetections = false;
    this.pcCustomHost = "";
    this.frameSequence = 0;

    // Models & Selection
    this.models = [];
    this.depthModels = [];
    this.selectedModel = null;      // YOLO model
    this.selectedDepthModel = null; // Depth model
    this.selectedResolution = 480;
    this.selectedDepthResolution = 512; // 512 Quality (Default) or 320 Speed
    this.confThreshold = 0.25;

    // Live display controls
    this.showDepthColormap = false;
    this.isInferenceRunning = false;
    this.animationFrameId = null;

    // FPS & Latency stats
    this.lastFrameTime = performance.now();
    this.cameraFps = 0;
    this.cameraFpsHistory = [];

    this.elements = {};
  }

  async init() {
    this.bindElements();
    this.attachEventListeners();
    this.camera = new MobileCameraManager(this.elements.cameraVideo);

    // Setup PC Client status listener
    pcClient.onStatusChangeCallback = (state, message) => this.handlePCStatusChange(state, message);

    await this.checkBackendStatus();
    await this.loadModelLists();
  }

  bindElements() {
    this.elements = {
      deviceRadios: document.querySelectorAll('input[name="inference-device"]'),
      appDeviceBadge: document.getElementById("app-device-badge"),
      serverDot: document.getElementById("server-dot"),
      serverText: document.getElementById("server-text"),
      setupScreen: document.getElementById("setup-screen"),
      liveScreen: document.getElementById("live-screen"),
      modelSelect: document.getElementById("model-select"),
      depthModelSelect: document.getElementById("depth-model-select"),
      resolutionSelect: document.getElementById("resolution-select"),
      confSlider: document.getElementById("conf-slider"),
      confValue: document.getElementById("conf-value"),
      phoneCacheCard: document.getElementById("phone-cache-card"),
      pcSettingsCard: document.getElementById("pc-settings-card"),
      pcRecordToggle: document.getElementById("pc-record-toggle"),
      pcRecordOptions: document.getElementById("pc-record-options"),
      pcRecordFreq: document.getElementById("pc-record-freq"),
      pcOnlyDetections: document.getElementById("pc-only-detections"),
      pcServerHost: document.getElementById("pc-server-host"),

      // Status info
      statusModelName: document.getElementById("status-model-name"),
      statusResolution: document.getElementById("status-resolution"),
      statusCacheState: document.getElementById("status-cache-state"),
      statusFileSize: document.getElementById("status-file-size"),

      statusDepthName: document.getElementById("status-depth-name"),
      statusDepthRes: document.getElementById("status-depth-res"),
      statusDepthSize: document.getElementById("status-depth-size"),
      statusDepthCacheState: document.getElementById("status-depth-cache-state"),

      btnDownload: document.getElementById("btn-download"),
      btnDownloadDepth: document.getElementById("btn-download-depth"),
      btnStart: document.getElementById("btn-start"),
      btnStartText: document.getElementById("btn-start-text"),
      btnStop: document.getElementById("btn-stop"),
      btnClearCache: document.getElementById("btn-clear-cache"),
      btnToggleDepth: document.getElementById("btn-toggle-depth"),

      downloadProgressContainer: document.getElementById("download-progress-container"),
      downloadFill: document.getElementById("download-fill"),
      downloadText: document.getElementById("download-text"),

      // Live elements
      cameraVideo: document.getElementById("camera-video"),
      depthCanvas: document.getElementById("depth-canvas"),
      overlayCanvas: document.getElementById("overlay-canvas"),
      navBanner: document.getElementById("nav-banner"),
      navBannerText: document.getElementById("nav-banner-text"),

      hudModelTag: document.getElementById("hud-model-tag"),
      hudProvider: document.getElementById("hud-provider"),
      hudRecBadge: document.getElementById("hud-rec-badge"),
      hudFps: document.getElementById("hud-fps"),
      hudObjects: document.getElementById("hud-objects"),

      hudYoloInf: document.getElementById("hud-yolo-inf"),
      hudDepthInf: document.getElementById("hud-depth-inf"),
      hudDepthFps: document.getElementById("hud-depth-fps"),
      hudDepthFpsBadge: document.getElementById("hud-depth-fps-badge"),
      hudSecLeft: document.getElementById("hud-sec-left"),
      hudSecCenter: document.getElementById("hud-sec-center"),
      hudSecRight: document.getElementById("hud-sec-right"),

      // Depth Resolution & Benchmark elements
      depthResolutionSelect: document.getElementById("depth-resolution-select"),
      depthResolutionGroup: document.getElementById("depth-resolution-group"),
      dbgDepthRes: document.getElementById("dbg-depth-res"),
      dbgDepthLat: document.getElementById("dbg-depth-lat"),
      dbgDepthFps: document.getElementById("dbg-depth-fps"),
      dbgE2eLat: document.getElementById("dbg-e2e-lat"),
      dbgLoadTime: document.getElementById("dbg-load-time"),
      dbgDropped: document.getElementById("dbg-dropped"),
      dbgProvider: document.getElementById("dbg-provider"),
      btnSwitchDepthRes: document.getElementById("btn-switch-depth-res"),
      btnSwitchDepthResText: document.getElementById("btn-switch-depth-res-text")
    };
  }

  attachEventListeners() {
    this.elements.deviceRadios.forEach(radio => {
      radio.addEventListener("change", (e) => this.onInferenceDeviceChanged(e.target.value));
    });

    this.elements.modelSelect.addEventListener("change", () => this.onYoloModelChanged());
    this.elements.depthModelSelect.addEventListener("change", () => this.onDepthModelChanged());
    this.elements.resolutionSelect.addEventListener("change", () => this.onResolutionChanged());

    if (this.elements.depthResolutionSelect) {
      this.elements.depthResolutionSelect.addEventListener("change", (e) => this.onDepthResolutionChanged(parseInt(e.target.value, 10)));
    }

    if (this.elements.btnSwitchDepthRes) {
      this.elements.btnSwitchDepthRes.addEventListener("click", () => this.switchDepthResolution());
    }

    this.elements.confSlider.addEventListener("input", (e) => {
      this.confThreshold = parseFloat(e.target.value);
      this.elements.confValue.textContent = this.confThreshold.toFixed(2);
    });

    if (this.elements.btnToggleDepth) {
      this.elements.btnToggleDepth.addEventListener("click", () => this.toggleDepthColormap());
    }

    if (this.elements.pcRecordToggle) {
      this.elements.pcRecordToggle.addEventListener("change", (e) => {
        this.pcRecordEnabled = e.target.checked;
        if (this.elements.pcRecordOptions) {
          this.elements.pcRecordOptions.style.display = this.pcRecordEnabled ? "block" : "none";
        }
      });
    }

    if (this.elements.pcRecordFreq) {
      this.elements.pcRecordFreq.addEventListener("change", (e) => {
        this.pcRecordFreq = parseInt(e.target.value, 10) || 5;
      });
    }

    if (this.elements.pcOnlyDetections) {
      this.elements.pcOnlyDetections.addEventListener("change", (e) => {
        this.pcOnlyDetections = e.target.checked;
      });
    }

    if (this.elements.pcServerHost) {
      this.elements.pcServerHost.addEventListener("input", (e) => {
        this.pcCustomHost = e.target.value.trim();
      });
    }

    this.elements.btnDownload.addEventListener("click", () => this.downloadSelectedYoloModel());
    this.elements.btnDownloadDepth.addEventListener("click", () => this.downloadSelectedDepthModel());
    this.elements.btnStart.addEventListener("click", () => this.startLiveDetection());
    this.elements.btnStop.addEventListener("click", () => this.stopLiveDetection());
    this.elements.btnClearCache.addEventListener("click", () => this.clearAllPhoneCache());
  }

  toggleDepthColormap() {
    this.showDepthColormap = !this.showDepthColormap;
    if (this.elements.btnToggleDepth) {
      this.elements.btnToggleDepth.classList.toggle("active", this.showDepthColormap);
      this.elements.btnToggleDepth.querySelector("span").textContent = this.showDepthColormap
        ? "🗺️ DEPTH: ON"
        : "🗺️ DEPTH: OFF";
    }
    if (this.elements.depthCanvas) {
      this.elements.depthCanvas.classList.toggle("hidden", !this.showDepthColormap);
    }
  }

  onInferenceDeviceChanged(mode) {
    this.inferenceMode = mode;
    console.log(`[ROD App] Switched inference mode to: ${mode.toUpperCase()}`);

    if (mode === "pc") {
      if (this.elements.appDeviceBadge) {
        this.elements.appDeviceBadge.textContent = "PC SERVER";
        this.elements.appDeviceBadge.style.background = "#059669";
      }
      this.elements.phoneCacheCard.classList.add("hidden");
      this.elements.pcSettingsCard.classList.remove("hidden");
      if (this.elements.btnStartText) {
        this.elements.btnStartText.textContent = "🚀 START PC STREAMING INFERENCE";
      }
      this.elements.btnStart.disabled = !this.selectedModel;
      pcClient.connect(this.pcCustomHost);
    } else {
      if (this.elements.appDeviceBadge) {
        this.elements.appDeviceBadge.textContent = "ON-DEVICE";
        this.elements.appDeviceBadge.style.background = "#0284c7";
      }
      this.elements.phoneCacheCard.classList.remove("hidden");
      this.elements.pcSettingsCard.classList.add("hidden");
      if (this.elements.btnStartText) {
        this.elements.btnStartText.textContent = "🚀 START LIVE DETECTION";
      }
      this.updateModelStatusDisplay();
    }
  }

  handlePCStatusChange(state, message) {
    console.log(`[PC Client Status] ${state}: ${message}`);
    if (this.inferenceMode === "pc") {
      if (state === "connected") {
        this.elements.serverDot.className = "status-dot online";
        this.elements.serverText.textContent = "PC Server Ready";
      } else if (state === "connecting") {
        this.elements.serverDot.className = "status-dot";
        this.elements.serverText.textContent = "Connecting to PC...";
      } else {
        this.elements.serverDot.className = "status-dot offline";
        this.elements.serverText.textContent = "PC Offline";
      }
    }
  }

  getYoloCacheKey() {
    if (!this.selectedModel) return "";
    return `${this.selectedModel.id}_${this.selectedResolution}`;
  }

  getDepthResolution() {
    if (!this.selectedDepthModel) return 512;
    if (this.selectedDepthModel.supported_resolutions && this.selectedDepthModel.supported_resolutions.length > 1) {
      return this.selectedDepthResolution || 512;
    }
    return this.selectedDepthModel.resolution || (this.selectedDepthModel.supported_resolutions ? this.selectedDepthModel.supported_resolutions[0] : 512);
  }

  getDepthCacheKey() {
    if (!this.selectedDepthModel) return "";
    const res = this.getDepthResolution();
    return `${this.selectedDepthModel.id}_${res}`;
  }

  async checkBackendStatus() {
    const health = await this.api.checkHealth();
    if (health.healthy) {
      this.elements.serverDot.className = "status-dot online";
      this.elements.serverText.textContent = `Online (${health.server_ip || "LAN"})`;
    } else {
      this.elements.serverDot.className = "status-dot offline";
      this.elements.serverText.textContent = "Offline (Local Cache)";
    }
  }

  async loadModelLists() {
    try {
      const allModels = await this.api.getModels();

      // Separate YOLO detectors and Depth models
      this.models = allModels.filter(m => m.task !== "depth" && m.category !== "depth_model");
      this.depthModels = allModels.filter(m => m.task === "depth" || m.category === "depth_model");

      // 1. Populate YOLO Dropdown
      this.elements.modelSelect.innerHTML = "";
      const categories = [
        { id: "converted_saved", label: "⚡ Converted Saved Models" },
        { id: "official_model", label: "📦 Official YOLO Models" }
      ];

      categories.forEach(cat => {
        const groupModels = this.models.filter(m => m.category === cat.id);
        if (groupModels.length > 0) {
          const optGroup = document.createElement("optgroup");
          optGroup.label = cat.label;
          groupModels.forEach(m => {
            const opt = document.createElement("option");
            opt.value = m.id;
            opt.textContent = `${m.name} (${m.file_size_mb} MB)`;
            optGroup.appendChild(opt);
          });
          this.elements.modelSelect.appendChild(optGroup);
        }
      });

      // 2. Populate Depth Dropdown
      this.elements.depthModelSelect.innerHTML = "";
      if (this.depthModels.length > 0) {
        this.depthModels.forEach(dm => {
          const opt = document.createElement("option");
          opt.value = dm.id;
          opt.textContent = `${dm.name} (${dm.file_size_mb} MB)`;
          this.elements.depthModelSelect.appendChild(opt);
        });
      } else {
        const opt = document.createElement("option");
        opt.value = "";
        opt.textContent = "No depth models available";
        this.elements.depthModelSelect.appendChild(opt);
      }

      // Default selections
      this.selectedModel = this.models[0] || null;
      this.selectedDepthModel = this.depthModels[0] || null;

      this.updateResolutionOptions();
      await this.updateModelStatusDisplay();
    } catch (err) {
      console.warn("Error loading models list:", err);
    }
  }

  updateResolutionOptions() {
    if (!this.selectedModel) return;
    const supported = (this.selectedModel.supported_resolutions && this.selectedModel.supported_resolutions.length > 0)
      ? this.selectedModel.supported_resolutions
      : [320, 480];
    const currentVal = parseInt(this.elements.resolutionSelect.value, 10);

    this.elements.resolutionSelect.innerHTML = "";
    supported.forEach(res => {
      const opt = document.createElement("option");
      opt.value = res;
      opt.textContent = `${res} × ${res}`;
      this.elements.resolutionSelect.appendChild(opt);
    });

    if (supported.includes(currentVal)) {
      this.elements.resolutionSelect.value = currentVal;
      this.selectedResolution = currentVal;
    } else {
      this.elements.resolutionSelect.value = supported[0];
      this.selectedResolution = supported[0];
    }
    this.elements.statusResolution.textContent = `${this.selectedResolution} × ${this.selectedResolution}`;
  }

  async onYoloModelChanged() {
    const id = this.elements.modelSelect.value;
    this.selectedModel = this.models.find(m => m.id === id) || null;
    this.updateResolutionOptions();
    await this.updateModelStatusDisplay();
  }

  async onDepthModelChanged() {
    const id = this.elements.depthModelSelect.value;
    this.selectedDepthModel = this.depthModels.find(m => m.id === id) || null;

    if (this.selectedDepthModel && this.selectedDepthModel.supported_resolutions && this.selectedDepthModel.supported_resolutions.length > 1) {
      if (this.elements.depthResolutionGroup) {
        this.elements.depthResolutionGroup.style.display = "block";
      }
      if (this.elements.depthResolutionSelect) {
        this.elements.depthResolutionSelect.value = String(this.selectedDepthResolution || 512);
      }
    } else {
      if (this.elements.depthResolutionGroup) {
        this.elements.depthResolutionGroup.style.display = "none";
      }
    }
    await this.updateModelStatusDisplay();
  }

  async onDepthResolutionChanged(newRes) {
    this.selectedDepthResolution = parseInt(newRes, 10) || 512;
    console.log(`[ROD App] Depth resolution set to: ${this.selectedDepthResolution}x${this.selectedDepthResolution}`);

    if (this.elements.depthResolutionSelect) {
      this.elements.depthResolutionSelect.value = String(this.selectedDepthResolution);
    }
    if (this.elements.btnSwitchDepthResText) {
      this.elements.btnSwitchDepthResText.textContent = `${this.selectedDepthResolution}p`;
    }
    await this.updateModelStatusDisplay();

    // If live inference is running on edge, reload depth model on the fly
    if (this.isInferenceRunning && this.inferenceMode === "edge" && this.selectedDepthModel) {
      await this.reloadDepthModelOnTheFly();
    }
  }

  async switchDepthResolution() {
    const nextRes = this.selectedDepthResolution === 512 ? 320 : 512;
    await this.onDepthResolutionChanged(nextRes);
  }

  async reloadDepthModelOnTheFly() {
    const depthKey = this.getDepthCacheKey();
    const cachedDepth = await modelCache.getModel(depthKey);
    const depthRes = this.getDepthResolution();
    if (cachedDepth) {
      let depthMeta = {};
      try {
        depthMeta = await this.api.getModelMetadata(this.selectedDepthModel.id, depthRes);
      } catch (e) {
        depthMeta = (cachedDepth && cachedDepth.metadata) || { id: this.selectedDepthModel.id, resolution: depthRes };
      }
      depthMeta.resolution = depthRes;
      await this.modelManager.loadDepthModel(cachedDepth.buffer, depthMeta);
      console.log(`[ROD App] Switched depth model to ${depthRes}x${depthRes} on the fly.`);
    } else {
      console.warn(`[ROD App] Depth model for resolution ${depthRes} not cached.`);
      alert(`Depth model for ${depthRes}×${depthRes} is not cached! Please download it from Setup first.`);
    }
  }

  async onResolutionChanged() {
    this.selectedResolution = parseInt(this.elements.resolutionSelect.value, 10) || 480;
    this.elements.statusResolution.textContent = `${this.selectedResolution} × ${this.selectedResolution}`;
    await this.updateModelStatusDisplay();
  }

  async updateModelStatusDisplay() {
    if (!this.selectedModel) {
      this.elements.btnStart.disabled = true;
      return;
    }

    if (this.inferenceMode === "pc") {
      this.elements.btnStart.disabled = false;
      return;
    }

    // Check YOLO model cache
    const yoloKey = this.getYoloCacheKey();
    this.elements.statusModelName.textContent = this.selectedModel.name || this.selectedModel.id;
    this.elements.statusResolution.textContent = `${this.selectedResolution} × ${this.selectedResolution}`;

    const isYoloCached = await modelCache.hasModel(yoloKey);
    if (isYoloCached) {
      this.elements.statusCacheState.textContent = "✓ Cached on Phone";
      this.elements.statusCacheState.style.color = "#10b981";
      this.elements.statusFileSize.textContent = "Cached";
      this.elements.btnDownload.textContent = "✓ YOLO Ready";
      this.elements.btnDownload.disabled = false;
    } else {
      this.elements.statusCacheState.textContent = "Not downloaded";
      this.elements.statusCacheState.style.color = "#f59e0b";
      this.elements.statusFileSize.textContent = `${this.selectedModel.file_size_mb} MB`;
      this.elements.btnDownload.textContent = `⬇ Download YOLO (${this.selectedModel.file_size_mb} MB)`;
      this.elements.btnDownload.disabled = false;
    }

    // Check Depth model cache
    let isDepthCached = false;
    if (this.selectedDepthModel) {
      const depthRes = this.getDepthResolution();
      const depthKey = this.getDepthCacheKey();
      this.elements.statusDepthName.textContent = this.selectedDepthModel.name || this.selectedDepthModel.id;
      if (this.elements.statusDepthRes) {
        this.elements.statusDepthRes.textContent = `${depthRes} × ${depthRes}`;
      }
      this.elements.statusDepthSize.textContent = `${this.selectedDepthModel.file_size_mb} MB`;

      isDepthCached = await modelCache.hasModel(depthKey);
      if (isDepthCached) {
        this.elements.statusDepthCacheState.textContent = `✓ Cached (${depthRes}p)`;
        this.elements.statusDepthCacheState.style.color = "#10b981";
        this.elements.btnDownloadDepth.textContent = `✓ Depth Ready (${depthRes}p)`;
        this.elements.btnDownloadDepth.disabled = false;
      } else {
        this.elements.statusDepthCacheState.textContent = `Not downloaded (${depthRes}p)`;
        this.elements.statusDepthCacheState.style.color = "#f59e0b";
        this.elements.btnDownloadDepth.textContent = `⬇ Download Depth ${depthRes}p (${this.selectedDepthModel.file_size_mb} MB)`;
        this.elements.btnDownloadDepth.disabled = false;
      }
    } else {
      this.elements.statusDepthName.textContent = "None Selected";
      this.elements.statusDepthCacheState.textContent = "Disabled";
    }

    // Enable Start button if YOLO is cached (Depth is optional/concurrent)
    this.elements.btnStart.disabled = !isYoloCached;
    this.elements.btnClearCache.style.display = (isYoloCached || isDepthCached) ? "block" : "none";
  }

  async downloadSelectedYoloModel() {
    if (!this.selectedModel) return;
    const modelId = this.selectedModel.id;
    const res = this.selectedResolution;
    const cacheKey = this.getYoloCacheKey();
    let cdnUrl = null;
    if (this.selectedModel.cdn_urls && this.selectedModel.cdn_urls[res]) {
      cdnUrl = this.selectedModel.cdn_urls[res];
    } else {
      cdnUrl = this.selectedModel.cdn_url || null;
    }

    await this.performDownload(modelId, res, cacheKey, this.elements.btnDownload, cdnUrl);
  }

  async downloadSelectedDepthModel() {
    if (!this.selectedDepthModel) return;
    const modelId = this.selectedDepthModel.id;
    const res = this.getDepthResolution();
    const cacheKey = this.getDepthCacheKey();
    let cdnUrl = null;
    if (this.selectedDepthModel.cdn_urls && this.selectedDepthModel.cdn_urls[res]) {
      cdnUrl = this.selectedDepthModel.cdn_urls[res];
    } else {
      cdnUrl = this.selectedDepthModel.cdn_url || null;
    }

    await this.performDownload(modelId, res, cacheKey, this.elements.btnDownloadDepth, cdnUrl);
  }

  async performDownload(modelId, resolution, cacheKey, buttonElement, cdnUrl = null) {
    this.elements.downloadProgressContainer.style.display = "block";
    this.elements.downloadFill.style.width = "0%";
    this.elements.downloadText.textContent = `Connecting for ${modelId}...`;
    buttonElement.disabled = true;

    try {
      const buffer = await this.api.downloadModelBinary(modelId, resolution, "onnx", (percent, loaded, total) => {
        const loadedMb = (loaded / (1024 * 1024)).toFixed(1);
        if (percent >= 0 && total > 0) {
          const totalMb = (total / (1024 * 1024)).toFixed(1);
          this.elements.downloadFill.style.width = `${percent}%`;
          this.elements.downloadText.textContent = `Downloading ${modelId}... ${percent}% (${loadedMb} / ${totalMb} MB)`;
        } else {
          this.elements.downloadFill.style.width = "70%";
          this.elements.downloadText.textContent = `Downloading ${modelId}... ${loadedMb} MB`;
        }
      }, 0, cdnUrl);

      this.elements.downloadText.textContent = `Saving ${modelId} to phone storage...`;

      let meta = {};
      try {
        meta = await this.api.getModelMetadata(modelId, resolution);
      } catch (_) {}

      await modelCache.saveModel(cacheKey, buffer, {
        modelId,
        resolution,
        version: "1.0",
        class_names: meta.class_names || (this.selectedModel && this.selectedModel.id === modelId ? this.selectedModel.class_names : []),
        output_shape: meta.output_shape || null,
        task: meta.task || (this.selectedModel && this.selectedModel.id === modelId ? this.selectedModel.task : "detection"),
        distance_heuristic: meta.distance_heuristic || null,
        input_tensor_name: meta.input_tensor_name || "images",
        output_tensor_name: meta.output_tensor_name || "output0"
      });

      this.elements.downloadText.textContent = `Ready: ${modelId} ✓`;
      await this.updateModelStatusDisplay();
    } catch (err) {
      alert(`Download error: ${err.message}`);
      this.elements.downloadText.textContent = `Failed: ${err.message}`;
    } finally {
      buttonElement.disabled = false;
      setTimeout(() => {
        this.elements.downloadProgressContainer.style.display = "none";
      }, 2500);
    }
  }

  async clearAllPhoneCache() {
    if (confirm("Clear all cached models from phone storage?")) {
      const list = await modelCache.listCachedModels();
      for (const item of list) {
        await modelCache.deleteModel(item.id);
      }
      await this.updateModelStatusDisplay();
    }
  }

  async startLiveDetection() {
    if (!this.selectedModel) return;

    try {
      this.elements.btnStart.disabled = true;
      if (this.inferenceMode === "edge") {
        await this.startPhoneEdgeMode();
      } else {
        await this.startPCServerMode();
      }
    } catch (err) {
      alert(`Error starting detection: ${err.message}`);
      console.error(err);
      this.stopLiveDetection();
    } finally {
      this.elements.btnStart.disabled = false;
    }
  }

  async startPhoneEdgeMode() {
    const yoloKey = this.getYoloCacheKey();
    if (this.elements.btnStartText) {
      this.elements.btnStartText.textContent = "⚡ Initializing ONNX runtime...";
    }

    // 1. Check YOLO cache
    const cachedYolo = await modelCache.getModel(yoloKey);
    if (!cachedYolo) {
      throw new Error("YOLO model is not cached. Please tap 'Download YOLO' first.");
    }
    let yoloMeta = {};
    try {
      yoloMeta = await this.api.getModelMetadata(this.selectedModel.id, this.selectedResolution);
    } catch (e) {
      yoloMeta = (cachedYolo && cachedYolo.metadata) || {
        id: this.selectedModel.id,
        resolution: this.selectedResolution,
        class_names: this.selectedModel.class_names || []
      };
    }
    if (!yoloMeta.class_names || yoloMeta.class_names.length === 0) {
      yoloMeta.class_names = (cachedYolo && cachedYolo.metadata && cachedYolo.metadata.class_names) || this.selectedModel.class_names || [];
    }

    // 2. Prepare parallel tasks for YOLO and Depth
    const loadTasks = [];

    loadTasks.push((async () => {
      if (this.elements.btnStartText) {
        this.elements.btnStartText.textContent = `⚡ Loading YOLO (${this.selectedResolution}x${this.selectedResolution})...`;
      }
      await this.modelManager.loadYoloModel(cachedYolo.buffer, yoloMeta);
      console.log(`[ROD App] YOLO model loaded successfully.`);
    })());

    if (this.selectedDepthModel) {
      const depthKey = this.getDepthCacheKey();
      const cachedDepth = await modelCache.getModel(depthKey);
      if (cachedDepth) {
        let depthMeta = {};
        const depthRes = this.getDepthResolution();
        try {
          depthMeta = await this.api.getModelMetadata(this.selectedDepthModel.id, depthRes);
        } catch (e) {
          depthMeta = (cachedDepth && cachedDepth.metadata) || { id: this.selectedDepthModel.id, resolution: depthRes };
        }
        depthMeta.resolution = depthRes;

        loadTasks.push((async () => {
          await this.modelManager.loadDepthModel(cachedDepth.buffer, depthMeta);
          console.log(`[ROD App] Depth model ${this.selectedDepthModel.id} (${depthRes}p) loaded successfully.`);
        })());
      } else {
        console.log("[ROD App] Depth model not cached; continuing with YOLO + bounding box heuristic.");
      }
    }

    // Load models concurrently in parallel
    await Promise.all(loadTasks);

    // 3. Start camera stream
    if (this.elements.btnStartText) {
      this.elements.btnStartText.textContent = "📷 Starting camera stream...";
    }
    await this.camera.start("environment");

    // 4. Switch screens
    this.elements.setupScreen.classList.add("hidden");
    this.elements.liveScreen.classList.remove("hidden");

    // 5. Update HUD tags
    const depthTag = this.modelManager.hasDepth() ? `+ DEPTH (${this.getDepthResolution()}p)` : "(BBox Dist)";
    this.elements.hudModelTag.textContent = `${this.selectedModel.id} ${depthTag}`;
    this.elements.hudProvider.textContent = this.modelManager.stats.yoloProvider.toUpperCase();

    this.isInferenceRunning = true;
    this.runPhonePipelineLoop();
  }

  async startPCServerMode() {
    if (this.elements.btnStartText) {
      this.elements.btnStartText.textContent = "Connecting to PC Server...";
    }

    if (!pcClient.isConnected) {
      pcClient.connect(this.pcCustomHost);
      let attempts = 0;
      while (!pcClient.isConnected && attempts < 15) {
        await new Promise(r => setTimeout(r, 200));
        attempts++;
      }
      if (!pcClient.isConnected) {
        throw new Error("Could not connect to PC WebSocket server.");
      }
    }

    const configured = pcClient.configure(
      this.selectedModel.id,
      this.selectedResolution,
      this.confThreshold,
      {
        enabled: this.pcRecordEnabled,
        frequency: this.pcRecordFreq,
        only_detections: this.pcOnlyDetections
      }
    );
    if (!configured) {
      throw new Error("Failed to configure YOLO session on PC server.");
    }

    pcClient.onResultCallback = (msg) => this.handlePCInferenceResult(msg);
    await this.camera.start("environment");

    this.elements.setupScreen.classList.add("hidden");
    this.elements.liveScreen.classList.remove("hidden");

    this.elements.hudModelTag.textContent = `${this.selectedModel.id} | PC`;
    this.elements.hudProvider.textContent = "PC (STREAMING)";
    if (this.elements.hudRecBadge) {
      this.elements.hudRecBadge.classList.toggle("hidden", !this.pcRecordEnabled);
    }

    this.frameSequence = 0;
    this.isInferenceRunning = true;
    this.runPCInferenceLoop();
  }

  stopLiveDetection() {
    this.isInferenceRunning = false;
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }

    if (this.camera) {
      this.camera.stop();
    }

    if (this.inferenceMode === "pc") {
      pcClient.stopSession();
    }

    const canvas = this.elements.overlayCanvas;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    this.temporalTracker.reset();

    this.elements.liveScreen.classList.add("hidden");
    this.elements.setupScreen.classList.remove("hidden");
  }

  /**
   * Phone Edge Continuous Pipeline Loop:
   * 1. Captures live camera square frames (~30 FPS) and letterbox frame for depth.
   * 2. Asynchronously triggers YOLO (~12 FPS) and Depth (~6–15 FPS).
   * 3. Performs Spatial Fusion (Median depth extraction + Sector hazards).
   * 4. Stabilizes tracks & smooths distance via TemporalTracker.
   * 5. Formulates Navigation Scene State.
   * 6. Renders bounding boxes, colormap overlay, and HUD telemetry.
   */
  runPhonePipelineLoop() {
    if (!this.isInferenceRunning) return;

    const yoloRes = this.selectedResolution || 480;

    const canvas = this.elements.overlayCanvas;
    const ctx = canvas.getContext("2d");

    if (canvas.width !== yoloRes || canvas.height !== yoloRes) {
      canvas.width = yoloRes;
      canvas.height = yoloRes;
    }

    const loop = async () => {
      if (!this.isInferenceRunning || this.inferenceMode !== "edge") return;

      const tStart = performance.now();

      // 1. Calculate camera FPS
      const now = performance.now();
      const delta = now - this.lastFrameTime;
      this.lastFrameTime = now;
      if (delta > 0) {
        this.cameraFpsHistory.push(1000 / delta);
        if (this.cameraFpsHistory.length > 10) this.cameraFpsHistory.shift();
        this.cameraFps = Number((this.cameraFpsHistory.reduce((a, b) => a + b, 0) / this.cameraFpsHistory.length).toFixed(1));
      }

      // 2. Capture latest 1:1 square frame for YOLO
      const yoloFrameData = this.camera.captureSquareFrame(yoloRes);

      if (yoloFrameData) {
        // Dispatches YOLO inference if worker is free and interval passed
        this.modelManager.maybeRunYolo(yoloFrameData, yoloRes, yoloRes, this.confThreshold);

        // Dispatches Depth inference with neutral gray letterboxing to preserve native 16:9 aspect ratio
        if (this.modelManager.hasDepth()) {
          const depthRes = this.getDepthResolution();
          const depthFrame = this.camera.captureLetterboxFrame(depthRes);
          if (depthFrame) {
            this.modelManager.maybeRunDepth(depthFrame.imageData, depthFrame.letterboxInfo);
          }
        }

        // 3. Spatial Fusion: Fuse latest YOLO detections with latest valid Depth Map
        const rawDetections = this.modelManager.latestDetections;
        const fusionResult = this.spatialFusion.fuse(
          rawDetections,
          this.modelManager.depth,
          yoloRes,
          yoloRes
        );

        // 4. Temporal Tracking: Smooth position & distance, retain lost tracks for 3 frames
        const stabilizedTracks = this.temporalTracker.update(fusionResult.fusedDetections);

        // 5. Navigation Scene State: Formulate scene summary & primary hazards
        const navState = this.navigationState.update(
          stabilizedTracks,
          fusionResult.sectors,
          yoloRes,
          yoloRes
        );

        // 6. Visual Rendering
        // A) Depth Colormap Overlay (if toggled)
        if (this.showDepthColormap && this.modelManager.hasDepth() && this.elements.depthCanvas) {
          this.modelManager.depth.renderColorMap(this.elements.depthCanvas, 0.70);
        }

        // B) Bounding Boxes & Distance Labels Overlay
        this.renderDetections(ctx, yoloRes, yoloRes, stabilizedTracks);

        // C) Navigation Banner & Hazards
        this.updateNavBanner(navState);

        // D) HUD Performance Dashboard
        const pipelineLat = Math.round(performance.now() - tStart);
        this.updateHUD(stabilizedTracks.length, pipelineLat, fusionResult.sectors);
      }

      this.animationFrameId = requestAnimationFrame(loop);
    };

    this.animationFrameId = requestAnimationFrame(loop);
  }

  updateNavBanner(navState) {
    if (!this.elements.navBanner || !this.elements.navBannerText) return;
    this.elements.navBannerText.textContent = navState.summaryText;

    const isHazard = navState.primaryHazard !== null;
    this.elements.navBanner.classList.toggle("hazard", isHazard);
  }

  updateHUD(objectsCount, pipelineLat, sectors) {
    this.elements.hudFps.textContent = this.cameraFps.toFixed(1);
    this.elements.hudObjects.textContent = objectsCount;

    if (this.elements.hudYoloInf) {
      if (this.modelManager.stats.lastError) {
        this.elements.hudYoloInf.textContent = "ERR!";
        this.elements.hudYoloInf.style.color = "#ef4444";
        this.elements.hudYoloInf.title = this.modelManager.stats.lastError;
      } else {
        this.elements.hudYoloInf.textContent = `${this.modelManager.stats.yoloLatencyMs}ms`;
        this.elements.hudYoloInf.style.color = "#38bdf8";
        this.elements.hudYoloInf.title = "";
      }
    }
    if (this.elements.hudDepthInf) {
      this.elements.hudDepthInf.textContent = this.modelManager.hasDepth()
        ? `${this.modelManager.stats.depthLatencyMs}ms`
        : "N/A";
    }
    if (this.elements.hudDepthFps) {
      if (this.modelManager.hasDepth()) {
        this.elements.hudDepthFps.textContent = this.modelManager.stats.depthFps > 0
          ? this.modelManager.stats.depthFps.toFixed(1)
          : "—";
        if (this.elements.hudDepthFpsBadge) this.elements.hudDepthFpsBadge.style.display = "inline";
      } else {
        if (this.elements.hudDepthFpsBadge) this.elements.hudDepthFpsBadge.style.display = "none";
      }
    }

    // Benchmark & Diagnostic HUD metrics
    if (this.elements.dbgDepthRes) {
      this.elements.dbgDepthRes.textContent = `${this.modelManager.stats.depthResolution || this.selectedDepthResolution}p`;
    }
    if (this.elements.dbgDepthLat) {
      this.elements.dbgDepthLat.textContent = `${this.modelManager.stats.depthLatencyMs}ms`;
    }
    if (this.elements.dbgDepthFps) {
      this.elements.dbgDepthFps.textContent = this.modelManager.stats.depthFps > 0
        ? this.modelManager.stats.depthFps.toFixed(1)
        : "—";
    }
    if (this.elements.dbgE2eLat) {
      this.elements.dbgE2eLat.textContent = `${this.modelManager.stats.e2eLatencyMs}ms`;
    }
    if (this.elements.dbgLoadTime) {
      this.elements.dbgLoadTime.textContent = `${this.modelManager.stats.depthLoadTimeMs}ms`;
    }
    if (this.elements.dbgDropped) {
      this.elements.dbgDropped.textContent = this.modelManager.stats.depthDroppedFrames;
    }
    if (this.elements.dbgProvider) {
      this.elements.dbgProvider.textContent = (this.modelManager.stats.depthProvider || "WASM").toUpperCase();
    }
    if (this.elements.btnSwitchDepthResText) {
      this.elements.btnSwitchDepthResText.textContent = `${this.modelManager.stats.depthResolution || this.selectedDepthResolution}p`;
    }

    if (sectors) {
      if (this.elements.hudSecLeft) this.elements.hudSecLeft.textContent = sectors.left.label;
      if (this.elements.hudSecCenter) this.elements.hudSecCenter.textContent = sectors.center.label;
      if (this.elements.hudSecRight) this.elements.hudSecRight.textContent = sectors.right.label;

      if (this.elements.hudSecCenter) {
        this.elements.hudSecCenter.style.color = sectors.center.has_obstacle ? "#ef4444" : "#f8fafc";
      }
  }

  /**
   * PC Server Inference Loop.
   */
  runPCInferenceLoop() {
    if (!this.isInferenceRunning) return;

    const loop = () => {
      if (!this.isInferenceRunning || this.inferenceMode !== "pc") return;

      const res = this.selectedResolution || 480;
      const canvas = this.elements.overlayCanvas;
      if (canvas.width !== res || canvas.height !== res) {
        canvas.width = res;
        canvas.height = res;
      }

      if (!pcClient.isAwaitingResponse) {
        const jpegBase64 = this.camera.captureSquareJpeg(res, 0.7);
        if (jpegBase64) {
          pcClient.sendFrame(this.frameSequence++, Date.now(), jpegBase64);
        }
      }

      this.animationFrameId = requestAnimationFrame(loop);
    };

    this.animationFrameId = requestAnimationFrame(loop);
  }

  handlePCInferenceResult(msg) {
    if (!this.isInferenceRunning) return;

    const res = this.selectedResolution || 480;
    const canvas = this.elements.overlayCanvas;
    const ctx = canvas.getContext("2d");

    if (canvas.width !== res || canvas.height !== res) {
      canvas.width = res;
      canvas.height = res;
    }

    this.renderDetections(ctx, res, res, msg.detections);

    this.elements.hudFps.textContent = msg.fps !== undefined ? msg.fps.toFixed(1) : "0.0";
    this.elements.hudObjects.textContent = msg.count !== undefined ? msg.count : (msg.detections ? msg.detections.length : 0);

    const lat = msg.latency || {};
    if (this.elements.hudYoloInf) this.elements.hudYoloInf.textContent = `${lat.inference_ms || 0}ms`;
    if (this.elements.hudDepthInf) this.elements.hudDepthInf.textContent = "N/A";
    if (this.elements.hudDepthFpsBadge) this.elements.hudDepthFpsBadge.style.display = "none";
  }

  /**
   * Universal 1:1 detection renderer for bounding boxes, class names, and metric distances.
   */
  renderDetections(ctx, width, height, detections) {
    ctx.clearRect(0, 0, width, height);
    if (!detections || detections.length === 0) return;

    const baseUnit = Math.max(0.7, width / 480);
    const fontSize = Math.max(12, Math.round(14 * baseUnit));
    ctx.font = `bold ${fontSize}px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif`;
    ctx.textBaseline = "top";

    const lineWidth = Math.max(2, Math.round(3 * baseUnit));
    const labelHeight = Math.round(24 * baseUnit);

    for (const det of detections) {
      const [x1, y1, x2, y2] = det.box;
      const boxW = Math.max(0, x2 - x1);
      const boxH = Math.max(0, y2 - y1);

      const className = det.className || det.class_name || "Object";
      const score = det.score !== undefined ? det.score : (det.confidence !== undefined ? det.confidence : 0);

      // Color palette based on hazard level
      let strokeColor = "#38bdf8";
      const dist = det.estimated_distance;
      if (dist !== null && dist !== undefined && dist <= 1.5) {
        strokeColor = "#ef4444"; // Dangerously close
      } else if (dist !== null && dist !== undefined && dist <= 3.0) {
        strokeColor = "#f59e0b"; // Caution
      } else {
        const lowerName = className.toLowerCase();
        if (lowerName.includes("stairs") || lowerName.includes("manhole")) {
          strokeColor = "#ef4444";
        } else if (lowerName.includes("car") || lowerName.includes("person") || lowerName.includes("bike")) {
          strokeColor = "#10b981";
        }
      }

      // Draw bounding box
      ctx.lineWidth = lineWidth;
      ctx.strokeStyle = strokeColor;
      ctx.strokeRect(x1, y1, boxW, boxH);

      // Label: CLASS SCORE • DISTANCE
      const distLabel = det.distance_label ? ` • ${det.distance_label}` : "";
      const labelText = `${className.toUpperCase()} ${(score * 100).toFixed(0)}%${distLabel}`;
      const textWidth = ctx.measureText(labelText).width;

      // Label background banner
      ctx.fillStyle = strokeColor;
      ctx.fillRect(x1, Math.max(0, y1 - labelHeight), textWidth + 12, labelHeight);

      // Label text
      ctx.fillStyle = "#000000";
      ctx.fillText(labelText, x1 + 6, Math.max(0, y1 - labelHeight) + 4);
    }
  }
}

window.addEventListener("DOMContentLoaded", () => {
  const app = new App();
  app.init();
});
