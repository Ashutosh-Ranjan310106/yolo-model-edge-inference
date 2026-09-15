import React, { useState, useEffect, useRef, useCallback } from "react";
import { Header } from "./components/Header.jsx";
import { HomeScreen } from "./components/HomeScreen.jsx";
import { SetupScreen } from "./components/SetupScreen.jsx";
import { LiveScreen } from "./components/LiveScreen.jsx";
import { YOLO_MODELS, DEPTH_MODELS } from "./config/modelCatalog.js";
import { modelCache } from "./core/cache.js";
import { MobileCameraManager } from "./core/camera.js";
import { SpatialDepthFusion } from "./core/spatialFusion.js";
import { TemporalTracker } from "./core/temporalTracker.js";
import { NavigationState } from "./core/navigationState.js";
import { ModelManager } from "./engine/modelManager.js";
import { downloadFromUrl, readFileAsArrayBuffer } from "./core/downloader.js";
import { renderDetections } from "./core/renderer.js";

export default function App() {
  const [activeScreen, setActiveScreen] = useState("home"); // "home" | "setup" | "live"
  const [selectedYolo, setSelectedYolo] = useState(YOLO_MODELS[0]);
  const [selectedDepth, setSelectedDepth] = useState(DEPTH_MODELS[0]);
  const [selectedResolution, setSelectedResolution] = useState(YOLO_MODELS[0]?.defaultResolution || 256);
  const [selectedDepthResolution, setSelectedDepthResolution] = useState(512);
  const [confThreshold, setConfThreshold] = useState(0.25);

  const [isYoloCached, setIsYoloCached] = useState(false);
  const [isDepthCached, setIsDepthCached] = useState(false);
  const [cachedCount, setCachedCount] = useState(0);

  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [downloadStatusText, setDownloadStatusText] = useState("");

  const [showDepthColormap, setShowDepthColormap] = useState(false);
  const [audioEnabled, setAudioEnabled] = useState(true);

  // Live Telemetry
  const [telemetry, setTelemetry] = useState({
    fps: 0,
    objectsCount: 0,
    yoloLatency: 0,
    depthLatency: 0,
    depthFps: 0,
    sectors: null,
    summaryText: "Path Clear",
    isHazard: false,
    provider: "WASM"
  });

  // DOM Refs
  const videoRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const depthCanvasRef = useRef(null);

  // Core Pipeline Instances
  const cameraRef = useRef(null);
  const modelManagerRef = useRef(null);
  const spatialFusionRef = useRef(null);
  const temporalTrackerRef = useRef(null);
  const navigationStateRef = useRef(null);
  const animFrameIdRef = useRef(null);
  const isRunningRef = useRef(false);
  const lastFrameTimeRef = useRef(performance.now());
  const fpsHistoryRef = useRef([]);

  // Initialize Core Engines
  useEffect(() => {
    modelManagerRef.current = new ModelManager();
    spatialFusionRef.current = new SpatialDepthFusion();
    temporalTrackerRef.current = new TemporalTracker();
    navigationStateRef.current = new NavigationState();

    return () => {
      if (animFrameIdRef.current) cancelAnimationFrame(animFrameIdRef.current);
      if (cameraRef.current) cameraRef.current.stop();
      if (modelManagerRef.current) modelManagerRef.current.dispose();
    };
  }, []);

  // Cache Keys
  const getYoloCacheKey = useCallback(() => {
    return `${selectedYolo.id}_${selectedResolution}`;
  }, [selectedYolo, selectedResolution]);

  const getDepthCacheKey = useCallback(() => {
    if (!selectedDepth || selectedDepth.id === "none") return "";
    return `${selectedDepth.id}_${selectedDepthResolution}`;
  }, [selectedDepth, selectedDepthResolution]);

  // Check Storage Status
  const checkCacheStatus = useCallback(async () => {
    const yoloKey = getYoloCacheKey();
    const hasYolo = await modelCache.hasModel(yoloKey);
    setIsYoloCached(hasYolo);

    const depthKey = getDepthCacheKey();
    if (depthKey) {
      const hasDepth = await modelCache.hasModel(depthKey);
      setIsDepthCached(hasDepth);
    } else {
      setIsDepthCached(false);
    }

    const list = await modelCache.listCachedModels();
    setCachedCount(list.length);
  }, [getYoloCacheKey, getDepthCacheKey]);

  useEffect(() => {
    checkCacheStatus();
  }, [checkCacheStatus]);

  // Model Selection Handlers
  const handleSelectYolo = (id) => {
    const model = YOLO_MODELS.find((m) => m.id === id);
    if (model) {
      setSelectedYolo(model);
      const supported = model.supportedResolutions || [320, 480];
      if (!supported.includes(selectedResolution)) {
        setSelectedResolution(supported[0]);
      }
    }
  };

  const handleSelectDepth = (id) => {
    if (id === "none") {
      setSelectedDepth({ id: "none", name: "None (BBox Heuristic)", approxSizeMb: 0 });
    } else {
      const model = DEPTH_MODELS.find((dm) => dm.id === id);
      if (model) {
        setSelectedDepth(model);
        const supported = model.supportedResolutions || [512, 320, 256];
        if (!supported.includes(selectedDepthResolution)) {
          setSelectedDepthResolution(model.defaultResolution || supported[0]);
        }
      }
    }
  };

  // Download Handlers
  const handleDownloadYolo = async () => {
    const cdnUrl = selectedYolo.cdnUrls?.[selectedResolution] || Object.values(selectedYolo.cdnUrls || {})[0];
    if (!cdnUrl) {
      alert("No CDN URL configured for this resolution.");
      return;
    }

    const cacheKey = getYoloCacheKey();
    setIsDownloading(true);
    setDownloadProgress(0);
    setDownloadStatusText(`Connecting to CDN for ${selectedYolo.name}...`);

    try {
      const buffer = await downloadFromUrl(cdnUrl, (percent, loaded, total) => {
        const loadedMb = (loaded / (1024 * 1024)).toFixed(1);
        if (total > 0) {
          const totalMb = (total / (1024 * 1024)).toFixed(1);
          setDownloadProgress(percent);
          setDownloadStatusText(`Downloading YOLO: ${percent}% (${loadedMb} / ${totalMb} MB)`);
        } else {
          setDownloadProgress(50);
          setDownloadStatusText(`Downloading YOLO: ${loadedMb} MB...`);
        }
      });

      setDownloadStatusText("Saving YOLO model to local storage...");
      await modelCache.saveModel(cacheKey, buffer, {
        id: selectedYolo.id,
        resolution: selectedResolution,
        classNames: selectedYolo.classNames,
        inputTensorName: selectedYolo.inputTensorName,
        outputTensorName: selectedYolo.outputTensorName
      });

      setDownloadStatusText("YOLO Model Ready! ✓");
      await checkCacheStatus();
    } catch (err) {
      alert(`Download failed: ${err.message}`);
      setDownloadStatusText(`Error: ${err.message}`);
    } finally {
      setTimeout(() => {
        setIsDownloading(false);
      }, 2000);
    }
  };

  const handleDownloadDepth = async () => {
    if (!selectedDepth || selectedDepth.id === "none") return;
    const cdnUrl = selectedDepth.cdnUrls?.[selectedDepthResolution] || Object.values(selectedDepth.cdnUrls || {})[0];
    if (!cdnUrl) {
      alert("No CDN URL configured for this depth resolution.");
      return;
    }

    const cacheKey = getDepthCacheKey();
    setIsDownloading(true);
    setDownloadProgress(0);
    setDownloadStatusText(`Connecting to CDN for ${selectedDepth.name}...`);

    try {
      const buffer = await downloadFromUrl(cdnUrl, (percent, loaded, total) => {
        const loadedMb = (loaded / (1024 * 1024)).toFixed(1);
        if (total > 0) {
          const totalMb = (total / (1024 * 1024)).toFixed(1);
          setDownloadProgress(percent);
          setDownloadStatusText(`Downloading Depth: ${percent}% (${loadedMb} / ${totalMb} MB)`);
        } else {
          setDownloadProgress(50);
          setDownloadStatusText(`Downloading Depth: ${loadedMb} MB...`);
        }
      });

      setDownloadStatusText("Saving Depth model to local storage...");
      await modelCache.saveModel(cacheKey, buffer, {
        id: selectedDepth.id,
        resolution: selectedDepthResolution,
        isMetric: selectedDepth.isMetric,
        inputTensorName: selectedDepth.inputTensorName,
        outputTensorName: selectedDepth.outputTensorName
      });

      setDownloadStatusText("Depth Model Ready! ✓");
      await checkCacheStatus();
    } catch (err) {
      alert(`Download failed: ${err.message}`);
      setDownloadStatusText(`Error: ${err.message}`);
    } finally {
      setTimeout(() => {
        setIsDownloading(false);
      }, 2000);
    }
  };

  // Custom Local File Upload
  const handleLoadLocalModel = async (file) => {
    const isDepth = file.name.toLowerCase().includes("depth");
    const targetModel = isDepth ? selectedDepth : selectedYolo;
    const cacheKey = isDepth ? getDepthCacheKey() : getYoloCacheKey();

    setIsDownloading(true);
    setDownloadProgress(0);
    setDownloadStatusText(`Loading ${file.name} from disk...`);

    try {
      const buffer = await readFileAsArrayBuffer(file, (percent) => {
        setDownloadProgress(percent);
        setDownloadStatusText(`Reading file: ${percent}%`);
      });

      await modelCache.saveModel(cacheKey, buffer, {
        id: targetModel.id,
        resolution: isDepth ? selectedDepthResolution : selectedResolution,
        classNames: selectedYolo.classNames,
        customFile: file.name
      });

      setDownloadStatusText(`Loaded ${file.name} successfully! ✓`);
      await checkCacheStatus();
    } catch (err) {
      alert(`Failed to load file: ${err.message}`);
    } finally {
      setTimeout(() => setIsDownloading(false), 2000);
    }
  };

  const handleClearCache = async () => {
    if (window.confirm("Clear all cached models from browser storage?")) {
      await modelCache.clearAll();
      await checkCacheStatus();
    }
  };

  const handleToggleAudio = () => {
    const next = !audioEnabled;
    setAudioEnabled(next);
    navigationStateRef.current?.setAudioEnabled(next);
  };

  // Switch Depth Resolution on the fly during live inference
  const handleSwitchDepthResolution = async () => {
    const nextRes = selectedDepthResolution === 512 ? 320 : 512;
    setSelectedDepthResolution(nextRes);

    if (isRunningRef.current && selectedDepth && selectedDepth.id !== "none") {
      const nextKey = `${selectedDepth.id}_${nextRes}`;
      const record = await modelCache.getModel(nextKey);
      if (record) {
        await modelManagerRef.current.loadDepthModel(record.buffer, {
          id: selectedDepth.id,
          resolution: nextRes,
          isMetric: selectedDepth.isMetric
        });
      } else {
        alert(`Depth model for ${nextRes}p is not cached. Please download it from Setup first.`);
      }
    }
  };

  // START LIVE DETECTION PIPELINE
  const handleStartDetection = async () => {
    const yoloKey = getYoloCacheKey();
    const yoloRecord = await modelCache.getModel(yoloKey);
    if (!yoloRecord) {
      alert("YOLO model is not cached. Please tap 'Get YOLO' first.");
      return;
    }

    try {
      // 1. Load YOLO model
      await modelManagerRef.current.loadYoloModel(yoloRecord.buffer, {
        id: selectedYolo.id,
        resolution: selectedResolution,
        classNames: selectedYolo.classNames,
        inputTensorName: selectedYolo.inputTensorName,
        outputTensorName: selectedYolo.outputTensorName
      });

      // 2. Load Depth model if cached
      if (selectedDepth && selectedDepth.id !== "none") {
        const depthKey = getDepthCacheKey();
        const depthRecord = await modelCache.getModel(depthKey);
        if (depthRecord) {
          await modelManagerRef.current.loadDepthModel(depthRecord.buffer, {
            id: selectedDepth.id,
            resolution: selectedDepthResolution,
            isMetric: selectedDepth.isMetric,
            inputTensorName: selectedDepth.inputTensorName,
            outputTensorName: selectedDepth.outputTensorName
          });
        }
      }

      // 3. Switch to Live Screen
      setActiveScreen("live");

      // Small tick to ensure video element is mounted in DOM
      setTimeout(async () => {
        if (!videoRef.current) return;
        cameraRef.current = new MobileCameraManager(videoRef.current);
        await cameraRef.current.start("environment");

        isRunningRef.current = true;
        runPipelineLoop();
      }, 100);
    } catch (err) {
      alert(`Error starting detection: ${err.message}`);
      console.error(err);
      handleStopDetection();
    }
  };

  // STOP LIVE DETECTION
  const handleStopDetection = () => {
    isRunningRef.current = false;
    if (animFrameIdRef.current) {
      cancelAnimationFrame(animFrameIdRef.current);
      animFrameIdRef.current = null;
    }

    if (cameraRef.current) {
      cameraRef.current.stop();
    }

    if (overlayCanvasRef.current) {
      const ctx = overlayCanvasRef.current.getContext("2d");
      ctx?.clearRect(0, 0, overlayCanvasRef.current.width, overlayCanvasRef.current.height);
    }

    temporalTrackerRef.current?.reset();
    setActiveScreen("setup");
  };

  // Continuous Detection Loop
  const runPipelineLoop = () => {
    if (!isRunningRef.current) return;

    const yoloRes = selectedResolution || 480;
    const overlayCanvas = overlayCanvasRef.current;
    if (overlayCanvas && (overlayCanvas.width !== yoloRes || overlayCanvas.height !== yoloRes)) {
      overlayCanvas.width = yoloRes;
      overlayCanvas.height = yoloRes;
    }

    const loop = async () => {
      if (!isRunningRef.current) return;

      // 1. Calculate camera sensor/pipeline FPS
      const now = performance.now();
      const delta = now - lastFrameTimeRef.current;
      lastFrameTimeRef.current = now;
      if (delta > 0) {
        fpsHistoryRef.current.push(1000 / delta);
        if (fpsHistoryRef.current.length > 10) fpsHistoryRef.current.shift();
      }
      const smoothFps = Number(
        (fpsHistoryRef.current.reduce((a, b) => a + b, 0) / (fpsHistoryRef.current.length || 1)).toFixed(1)
      );

      // 2. Capture 1:1 square frame for YOLO
      if (cameraRef.current) {
        const yoloFrame = cameraRef.current.captureSquareFrame(yoloRes);
        if (yoloFrame) {
          // Asynchronously trigger YOLO
          modelManagerRef.current?.maybeRunYolo(yoloFrame, yoloRes, yoloRes, confThreshold);

          // Asynchronously trigger Depth with aspect-ratio letterbox padding
          if (modelManagerRef.current?.hasDepth()) {
            const depthRes = selectedDepthResolution || 512;
            const depthFrame = cameraRef.current.captureLetterboxFrame(depthRes);
            if (depthFrame) {
              modelManagerRef.current.maybeRunDepth(depthFrame.imageData, depthFrame.letterboxInfo);
            }
          }

          // 3. Spatial Fusion: Fuse 2D boxes with depth map
          const rawDetections = modelManagerRef.current?.latestDetections || [];
          const fusionResult = spatialFusionRef.current.fuse(
            rawDetections,
            modelManagerRef.current?.depth,
            yoloRes,
            yoloRes
          );

          // 4. Temporal Multi-Object Tracking & Smoothing
          const stabilizedTracks = temporalTrackerRef.current.update(fusionResult.fusedDetections);

          // 5. Navigation Scene Evaluation & Voice Guidance
          const navState = navigationStateRef.current.update(stabilizedTracks, fusionResult.sectors);

          // 6. Visual Rendering
          // A) Depth Colormap Canvas
          if (showDepthColormap && modelManagerRef.current?.hasDepth() && depthCanvasRef.current) {
            modelManagerRef.current.depth.renderColorMap(depthCanvasRef.current, 0.70);
          }

          // B) Bounding boxes & corridor
          if (overlayCanvasRef.current) {
            const ctx = overlayCanvasRef.current.getContext("2d");
            if (ctx) {
              renderDetections(ctx, yoloRes, yoloRes, stabilizedTracks);
            }
          }

          // Update Telemetry state
          const stats = modelManagerRef.current?.stats || {};
          setTelemetry({
            fps: smoothFps,
            objectsCount: stabilizedTracks.length,
            yoloLatency: stats.yoloLatencyMs || 0,
            depthLatency: stats.depthLatencyMs || 0,
            depthFps: stats.depthFps || 0,
            sectors: fusionResult.sectors,
            summaryText: navState.summaryText,
            isHazard: navState.primaryHazard !== null,
            provider: (stats.yoloProvider || "WASM").toUpperCase()
          });
        }
      }

      animFrameIdRef.current = requestAnimationFrame(loop);
    };

    animFrameIdRef.current = requestAnimationFrame(loop);
  };

  return (
    <>
      <Header 
        currentScreen={activeScreen} 
        onGoHome={() => setActiveScreen("home")} 
        cachedCount={cachedCount} 
      />

      {activeScreen === "home" ? (
        <HomeScreen onNavigateToNav={() => setActiveScreen("setup")} />
      ) : activeScreen === "setup" ? (
        <SetupScreen
          yoloModels={YOLO_MODELS}
          depthModels={DEPTH_MODELS}
          selectedYolo={selectedYolo}
          selectedDepth={selectedDepth}
          selectedResolution={selectedResolution}
          selectedDepthResolution={selectedDepthResolution}
          confThreshold={confThreshold}
          isYoloCached={isYoloCached}
          isDepthCached={isDepthCached}
          downloadProgress={downloadProgress}
          downloadStatusText={downloadStatusText}
          isDownloading={isDownloading}
          onSelectYolo={handleSelectYolo}
          onSelectDepth={handleSelectDepth}
          onSelectResolution={setSelectedResolution}
          onSelectDepthResolution={setSelectedDepthResolution}
          onChangeConfThreshold={setConfThreshold}
          onDownloadYolo={handleDownloadYolo}
          onDownloadDepth={handleDownloadDepth}
          onLoadLocalModel={handleLoadLocalModel}
          onClearCache={handleClearCache}
          onStartDetection={handleStartDetection}
          onBackToHome={() => setActiveScreen("home")}
        />
      ) : (
        <LiveScreen
          videoRef={videoRef}
          overlayCanvasRef={overlayCanvasRef}
          depthCanvasRef={depthCanvasRef}
          summaryText={telemetry.summaryText}
          isHazard={telemetry.isHazard}
          audioEnabled={audioEnabled}
          onToggleAudio={handleToggleAudio}
          fps={telemetry.fps}
          objectsCount={telemetry.objectsCount}
          yoloLatency={telemetry.yoloLatency}
          depthLatency={telemetry.depthLatency}
          depthFps={telemetry.depthFps}
          hasDepth={modelManagerRef.current?.hasDepth()}
          modelName={selectedYolo.name.split("(")[0]}
          sectors={telemetry.sectors}
          provider={telemetry.provider}
          showDepthColormap={showDepthColormap}
          onToggleDepthColormap={() => setShowDepthColormap((prev) => !prev)}
          depthResolution={selectedDepthResolution}
          onSwitchDepthResolution={handleSwitchDepthResolution}
          onStop={handleStopDetection}
        />
      )}
    </>
  );
}
