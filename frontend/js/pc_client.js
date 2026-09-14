/**
 * PC / Server Mode WebSocket Client for real-time YOLO inference.
 * Implements a non-blocking 'Latest Frame' pipeline to strictly avoid frame backlog.
 */

export class PCInferenceClient {
  constructor() {
    this.ws = null;
    this.isConnected = false;
    this.isAwaitingResponse = false;
    this.onResultCallback = null;
    this.onStatusChangeCallback = null;
    this.reconnectTimer = null;
    this.serverUrl = "";
  }

  getWebSocketUrl(customHost = "") {
    if (customHost) {
      const clean = customHost.replace(/^https?:\/\//, "").replace(/^wss?:\/\//, "").replace(/\/+$/, "");
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      return `${protocol}//${clean}/ws/inference`;
    }
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${protocol}//${window.location.host}/ws/inference`;
  }

  connect(customHost = "") {
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    const wsUrl = this.getWebSocketUrl(customHost);
    this.serverUrl = wsUrl;
    console.log(`[PC Client] Connecting to WebSocket: ${wsUrl}`);
    this.notifyStatus("connecting", "Connecting to PC Server...");

    try {
      this.ws = new WebSocket(wsUrl);
    } catch (err) {
      console.error("[PC Client] WebSocket creation failed:", err);
      this.notifyStatus("disconnected", "Connection failed");
      return;
    }

    this.ws.onopen = () => {
      console.log("[PC Client] WebSocket connected successfully.");
      this.isConnected = true;
      this.isAwaitingResponse = false;
      this.notifyStatus("connected", "Connected to PC Server");
    };

    this.ws.onmessage = (event) => {
      this.isAwaitingResponse = false; // PC finished processing, ready for newest frame
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === "result" && this.onResultCallback) {
          this.onResultCallback(msg);
        } else if (msg.type === "configured") {
          console.log("[PC Client] Configured on PC:", msg);
        } else if (msg.type === "session_ended") {
          console.log("[PC Client] Session saved on PC:", msg.summary);
        }
      } catch (e) {
        console.error("[PC Client] Message parse error:", e);
      }
    };

    this.ws.onerror = (err) => {
      console.warn("[PC Client] WebSocket error:", err);
    };

    this.ws.onclose = () => {
      console.log("[PC Client] WebSocket disconnected.");
      this.isConnected = false;
      this.isAwaitingResponse = false;
      this.notifyStatus("disconnected", "Disconnected from PC Server");
    };
  }

  disconnect() {
    if (this.ws) {
      try {
        this.ws.close();
      } catch {}
      this.ws = null;
    }
    this.isConnected = false;
    this.isAwaitingResponse = false;
  }

  notifyStatus(state, message) {
    if (this.onStatusChangeCallback) {
      this.onStatusChangeCallback(state, message);
    }
  }

  configure(modelName, resolution = 480, confidence = 0.25, recordConfig = null, depthModelName = "yolo26n_depth", depthResolution = 512, enableDepth = true) {
    if (!this.isConnected || !this.ws) {
      return false;
    }
    const payload = {
      type: "configure",
      model: modelName,
      resolution: resolution,
      confidence: confidence,
      depth_model: depthModelName,
      depth_resolution: depthResolution,
      enable_depth: enableDepth,
      record: recordConfig || { enabled: false, frequency: 5, only_detections: false }
    };
    this.ws.send(JSON.stringify(payload));
    return true;
  }

  /**
   * Transmits a camera frame ONLY if PC is idle (no queued backlog).
   * If PC is busy with previous frame, returns false immediately (frame dropped).
   */
  sendFrame(frameId, timestamp, jpegBase64) {
    if (!this.isConnected || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return false;
    }

    if (this.isAwaitingResponse) {
      // Drop frame to preserve real-time responsiveness
      return false;
    }

    this.isAwaitingResponse = true;
    const payload = {
      type: "frame",
      frame_id: frameId,
      timestamp: timestamp,
      image: jpegBase64
    };

    this.ws.send(JSON.stringify(payload));
    return true;
  }

  stopSession() {
    if (this.isConnected && this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "stop_session" }));
    }
  }
}

export const pcClient = new PCInferenceClient();
