import React from "react";
import { NavigationBanner } from "./NavigationBanner.jsx";
import { HudDashboard } from "./HudDashboard.jsx";
import { Layers, SquareSlash, Zap } from "lucide-react";

export function LiveScreen({
  videoRef,
  overlayCanvasRef,
  depthCanvasRef,
  summaryText,
  isHazard,
  audioEnabled,
  onToggleAudio,
  fps,
  objectsCount,
  yoloLatency,
  depthLatency,
  depthFps,
  hasDepth,
  modelName,
  sectors,
  provider,
  showDepthColormap,
  onToggleDepthColormap,
  depthResolution,
  onSwitchDepthResolution,
  onStop
}) {
  return (
    <section className="screen live-screen">
      {/* Top Voice & Navigation Banner */}
      <NavigationBanner
        summaryText={summaryText}
        isHazard={isHazard}
        audioEnabled={audioEnabled}
        onToggleAudio={onToggleAudio}
      />

      {/* Floating HUD Dashboard */}
      <HudDashboard
        fps={fps}
        objectsCount={objectsCount}
        yoloLatency={yoloLatency}
        depthLatency={depthLatency}
        depthFps={depthFps}
        hasDepth={hasDepth}
        modelName={modelName}
        sectors={sectors}
        provider={provider}
      />

      {/* 1:1 Center-Cropped Viewport */}
      <div className="camera-viewport-1to1">
        <video ref={videoRef} playsInline muted autoPlay />
        <canvas
          ref={depthCanvasRef}
          className={`depth-canvas ${showDepthColormap ? "" : "hidden"}`}
        />
        <canvas ref={overlayCanvasRef} className="overlay-canvas" />
        <div className="viewport-badge">1:1 SQUARE</div>
      </div>

      {/* Bottom Floating Control Bar */}
      <div className="hud-bottom-bar">
        {hasDepth && (
          <>
            <button
              className="btn-toggle-depth"
              style={{ background: "rgba(59, 130, 246, 0.25)", borderColor: "#3b82f6" }}
              onClick={onSwitchDepthResolution}
            >
              <Zap size={14} />
              <span>{depthResolution}p</span>
            </button>

            <button
              className={`btn-toggle-depth ${showDepthColormap ? "active" : ""}`}
              onClick={onToggleDepthColormap}
            >
              <Layers size={14} />
              <span>{showDepthColormap ? "DEPTH: ON" : "DEPTH: OFF"}</span>
            </button>
          </>
        )}

        <button className="btn-stop-camera" onClick={onStop}>
          <SquareSlash size={16} />
          <span>STOP DETECTION</span>
        </button>
      </div>
    </section>
  );
}
