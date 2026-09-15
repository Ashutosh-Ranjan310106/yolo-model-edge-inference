import React from "react";

export function HudDashboard({
  fps = 0,
  objectsCount = 0,
  yoloLatency = 0,
  depthLatency = 0,
  depthFps = 0,
  hasDepth = false,
  modelName = "",
  sectors = null,
  provider = "WASM"
}) {
  return (
    <div className="hud-dashboard">
      <div className="hud-pill">
        <div className="hud-model-title">
          <span className="metric-highlight">{modelName}</span>
          <span className="hud-provider-tag">[{provider}]</span>
        </div>
        <div className="hud-stats-group">
          <span>FPS: <b className="metric-highlight">{fps.toFixed(1)}</b></span>
          <span>Objects: <b className="metric-highlight">{objectsCount}</b></span>
        </div>
      </div>

      <div className="hud-latency-pill">
        <span>YOLO: <b style={{ color: "#38bdf8" }}>{yoloLatency}ms</b></span>
        <span>
          Depth: <b>{hasDepth ? `${depthLatency}ms` : "N/A"}</b>
          {hasDepth && depthFps > 0 && (
            <span style={{ opacity: 0.8, marginLeft: 4 }}>({depthFps.toFixed(1)} fps)</span>
          )}
        </span>
        {sectors && (
          <span className="hud-sectors">
            L:<b>{sectors.left?.label || "—"}</b>{" "}
            C:<b style={{ color: sectors.center?.hasObstacle ? "#ef4444" : "#f8fafc" }}>
              {sectors.center?.label || "—"}
            </b>{" "}
            R:<b>{sectors.right?.label || "—"}</b>
          </span>
        )}
      </div>
    </div>
  );
}
