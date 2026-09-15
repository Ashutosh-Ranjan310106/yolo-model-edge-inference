import React, { useRef } from "react";
import { Download, CheckCircle, Trash2, Upload, Play, RefreshCw, ChevronLeft } from "lucide-react";

export function SetupScreen({
  yoloModels,
  depthModels,
  selectedYolo,
  selectedDepth,
  selectedResolution,
  selectedDepthResolution,
  confThreshold,
  isYoloCached,
  isDepthCached,
  downloadProgress,
  downloadStatusText,
  isDownloading,
  onSelectYolo,
  onSelectDepth,
  onSelectResolution,
  onSelectDepthResolution,
  onChangeConfThreshold,
  onDownloadYolo,
  onDownloadDepth,
  onLoadLocalModel,
  onClearCache,
  onStartDetection,
  onBackToHome
}) {
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      onLoadLocalModel(file);
    }
  };

  const supportedResolutions = selectedYolo?.supportedResolutions || [320, 480];

  return (
    <main className="screen setup-screen">
      {onBackToHome && (
        <button className="btn-back-home" onClick={onBackToHome}>
          <ChevronLeft size={16} />
          <span>Back to VisionX Home</span>
        </button>
      )}

      {/* Model & Parameter Selection Card */}
      <div className="card">
        <h2 className="card-title">⚙ MODEL & RESOLUTION</h2>

        <div className="form-group">
          <label htmlFor="yolo-select">YOLO Object Detector</label>
          <select
            id="yolo-select"
            value={selectedYolo?.id || ""}
            onChange={(e) => onSelectYolo(e.target.value)}
          >
            {yoloModels.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name} (~{m.approxSizeMb} MB)
              </option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label htmlFor="depth-select">Depth Estimation Model</label>
          <select
            id="depth-select"
            value={selectedDepth?.id || "none"}
            onChange={(e) => onSelectDepth(e.target.value)}
          >
            <option value="none">None (Use Bounding-Box Area Heuristic)</option>
            {depthModels.map((dm) => (
              <option key={dm.id} value={dm.id}>
                {dm.name} (~{dm.approxSizeMb} MB)
              </option>
            ))}
          </select>
        </div>

        {selectedDepth && selectedDepth.id !== "none" && (
          <div className="form-group">
            <label htmlFor="depth-res-select">Depth Resolution</label>
            <select
              id="depth-res-select"
              value={selectedDepthResolution}
              onChange={(e) => onSelectDepthResolution(parseInt(e.target.value, 10))}
            >
              {(selectedDepth.supportedResolutions || [512, 320, 256]).map((res) => (
                <option key={res} value={res}>
                  {res} × {res} {res === 512 ? "(Quality Mode — Default)" : res === 320 ? "(Fast / Battery-Saver)" : res === 256 ? "(Ultra Fast / Low Power)" : ""}
                </option>
              ))}
            </select>
          </div>
        )}

        <div className="form-group">
          <label htmlFor="res-select">YOLO Input Resolution</label>
          <select
            id="res-select"
            value={selectedResolution}
            onChange={(e) => onSelectResolution(parseInt(e.target.value, 10))}
          >
            {supportedResolutions.map((res) => (
              <option key={res} value={res}>
                {res} × {res}
              </option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label htmlFor="conf-slider">
            Confidence Threshold: <span className="slider-val">{confThreshold.toFixed(2)}</span>
          </label>
          <div className="slider-container">
            <input
              type="range"
              id="conf-slider"
              min="0.10"
              max="0.90"
              step="0.05"
              value={confThreshold}
              onChange={(e) => onChangeConfThreshold(parseFloat(e.target.value))}
            />
          </div>
        </div>
      </div>

      {/* On-Device Storage & Local Files Card */}
      <div className="card">
        <h2 className="card-title">📦 ON-DEVICE MODEL STORAGE</h2>

        <div className="model-status-box">
          {/* YOLO Status */}
          <div className="status-row">
            <span>YOLO Model:</span>
            <b>{selectedYolo?.name?.split("(")[0] || selectedYolo?.id}</b>
          </div>
          <div className="status-row">
            <span>Resolution:</span>
            <b>{selectedResolution} × {selectedResolution}</b>
          </div>
          <div className="status-row">
            <span>YOLO Storage:</span>
            <b style={{ color: isYoloCached ? "#10b981" : "#f59e0b" }}>
              {isYoloCached ? "✓ Ready in Storage" : "Not Downloaded"}
            </b>
          </div>

          {/* Depth Status */}
          {selectedDepth && selectedDepth.id !== "none" && (
            <div style={{ marginTop: 10, paddingTop: 10, borderTop: "1px solid rgba(255,255,255,0.08)" }}>
              <div className="status-row">
                <span>Depth Model:</span>
                <b>{selectedDepth?.name?.split("(")[0] || selectedDepth?.id}</b>
              </div>
              <div className="status-row">
                <span>Depth Resolution:</span>
                <b>{selectedDepthResolution} × {selectedDepthResolution}</b>
              </div>
              <div className="status-row">
                <span>Depth Storage:</span>
                <b style={{ color: isDepthCached ? "#10b981" : "#f59e0b" }}>
                  {isDepthCached ? "✓ Ready in Storage" : "Not Downloaded"}
                </b>
              </div>
            </div>
          )}

          {/* Progress Bar */}
          {isDownloading && (
            <div className="progress-container">
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${downloadProgress}%` }}></div>
              </div>
              <div className="progress-text">{downloadStatusText}</div>
            </div>
          )}
        </div>

        {/* Action Buttons */}
        <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
          <button
            className={`btn ${isYoloCached ? "btn-ready" : "btn-secondary"}`}
            style={{ flex: 1 }}
            onClick={onDownloadYolo}
            disabled={isDownloading}
          >
            {isYoloCached ? <CheckCircle size={15} /> : <Download size={15} />}
            <span>{isYoloCached ? "YOLO Ready" : `Get YOLO (${selectedYolo?.approxSizeMb} MB)`}</span>
          </button>

          {selectedDepth && selectedDepth.id !== "none" && (
            <button
              className={`btn ${isDepthCached ? "btn-ready" : "btn-secondary"}`}
              style={{ flex: 1 }}
              onClick={onDownloadDepth}
              disabled={isDownloading}
            >
              {isDepthCached ? <CheckCircle size={15} /> : <Download size={15} />}
              <span>{isDepthCached ? "Depth Ready" : `Get Depth (${selectedDepth?.approxSizeMb} MB)`}</span>
            </button>
          )}
        </div>

        {/* Local File Upload Option */}
        <div style={{ marginTop: 10 }}>
          <input
            type="file"
            accept=".onnx"
            ref={fileInputRef}
            style={{ display: "none" }}
            onChange={handleFileChange}
          />
          <button
            className="btn btn-outline"
            style={{ width: "100%" }}
            onClick={() => fileInputRef.current?.click()}
            title="Pick a custom .onnx model directly from your phone or PC disk"
          >
            <Upload size={14} />
            <span>Load Custom .onnx from Device Disk</span>
          </button>
        </div>

        {(isYoloCached || isDepthCached) && (
          <button
            className="btn btn-danger"
            style={{ marginTop: 10, width: "100%" }}
            onClick={onClearCache}
          >
            <Trash2 size={14} />
            <span>Clear Storage Cache</span>
          </button>
        )}
      </div>

      {/* Start Button */}
      <button
        className="btn btn-primary"
        disabled={!isYoloCached || isDownloading}
        onClick={onStartDetection}
      >
        <Play size={18} />
        <span>START LIVE DETECTION</span>
      </button>
    </main>
  );
}
