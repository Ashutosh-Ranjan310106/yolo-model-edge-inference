/**
 * API Client for the ROD Model Registry Backend.
 * Features chunked Range download (2MB chunks) to prevent DevTunnel,
 * proxy, and mobile gateway timeouts on large neural network models.
 */
export class ModelRegistryAPI {
  constructor(baseUrl = "") {
    this.baseUrl = baseUrl || window.location.origin;
  }

  setBaseUrl(url) {
    this.baseUrl = url.replace(/\/+$/, "");
  }

  async checkHealth() {
    try {
      const res = await fetch(`${this.baseUrl}/api/health`, { cache: "no-store" });
      if (!res.ok) return { healthy: false, error: `HTTP ${res.status}` };
      const data = await res.json();
      return { healthy: true, ...data };
    } catch (err) {
      return { healthy: false, error: err.message };
    }
  }

  async getModels() {
    const res = await fetch(`${this.baseUrl}/api/models`, { cache: "no-store" });
    if (!res.ok) throw new Error(`Failed to fetch models: HTTP ${res.status}`);
    const data = await res.json();
    return data.models || [];
  }

  async getModelMetadata(modelId, resolution = 480) {
    const res = await fetch(`${this.baseUrl}/api/models/${encodeURIComponent(modelId)}/metadata?resolution=${resolution}`);
    if (!res.ok) throw new Error(`Failed to fetch metadata: HTTP ${res.status}`);
    return await res.json();
  }

  async getModelStatus(modelId, resolution = 480, format = "onnx") {
    try {
      const res = await fetch(`${this.baseUrl}/api/models/${encodeURIComponent(modelId)}/status?resolution=${resolution}&format=${format}`, { cache: "no-store" });
      if (!res.ok) return { status: "not_prepared" };
      return await res.json();
    } catch {
      return { status: "not_prepared" };
    }
  }

  /**
  /**
   * Download model binary using high-speed streaming reader (ReadableStream).
   * Provides real-time progress callbacks and prevents browser HTTP cache freezing.
   * @param {string} modelId
   * @param {number} resolution
   * @param {string} format
   * @param {Function} onProgress - (percent, loadedBytes, totalBytes) => void
   * @param {number} expectedSizeBytes - optional expected size in bytes
   * @param {string} cdnUrl - optional direct GitHub/Cloudflare CDN URL
   * @returns {Promise<ArrayBuffer>}
   */
  async downloadModelBinary(modelId, resolution = 480, format = "onnx", onProgress = null, expectedSizeBytes = 0, cdnUrl = null) {
    const timestamp = Date.now();
    const localUrl = `${this.baseUrl}/api/models/${encodeURIComponent(modelId)}/download?resolution=${resolution}&format=${format}&_t=${timestamp}`;

    const PUBLIC_MIRRORS = {
      "yolo26s_depth": "https://huggingface.co/AXERA-TECH/Yolo26-Depth/resolve/main/onnx/yolo26s-depth.onnx",
      "yolo26n_depth": "https://huggingface.co/AXERA-TECH/Yolo26-Depth/resolve/main/onnx/yolo26n-depth.onnx",
      "depth_anything_v2_small_quantized": "https://huggingface.co/onnx-community/Depth-Anything-V2-Small-ONNX/resolve/main/onnx/model_quantized.onnx",
      "depth_anything_v2_metric_small": "https://huggingface.co/onnx-community/Depth-Anything-V2-Small-ONNX/resolve/main/onnx/model.onnx"
    };

    // Candidates in priority order: GitHub CDN -> Public HF Edge CDN -> Local PC Backend
    const candidateUrls = [];
    if (cdnUrl) candidateUrls.push(cdnUrl);
    if (PUBLIC_MIRRORS[modelId] && PUBLIC_MIRRORS[modelId] !== cdnUrl) {
      candidateUrls.push(PUBLIC_MIRRORS[modelId]);
    }
    candidateUrls.push(localUrl);

    let downloadUrl = localUrl;
    let totalBytes = expectedSizeBytes || 0;
    let supportsRange = false;

    // Probe candidate URLs to find the fastest live CDN
    for (const url of candidateUrls) {
      try {
        const probeUrl = url.includes("?") ? `${url}&_t=${timestamp}` : `${url}?_t=${timestamp}`;
        const probeRes = await fetch(probeUrl, {
          headers: { "Range": "bytes=0-0" },
          cache: "no-store"
        });
        if (probeRes.ok || probeRes.status === 206) {
          downloadUrl = probeUrl;
          supportsRange = (probeRes.status === 206);
          const cr = probeRes.headers.get("content-range");
          if (cr && cr.includes("/")) {
            totalBytes = parseInt(cr.split("/")[1], 10);
          }
          console.log(`[API] 🚀 Selected live fast source: ${downloadUrl.startsWith("http://localhost") || downloadUrl.startsWith("http://127") ? "Local Backend" : downloadUrl.substring(0, 45) + "..."}`);
          break;
        }
      } catch (_) {
        // Probe failed, try next candidate
      }
    }

    // 2. For models > 15MB over tunnels/LAN, accelerate with 3 parallel streams to bypass single-connection bandwidth limits
    const numWorkers = (supportsRange && totalBytes > 15 * 1024 * 1024) ? 3 : 1;

    if (numWorkers > 1) {
      console.log(`[API] Accelerating ${(totalBytes / (1024 * 1024)).toFixed(1)} MB model with ${numWorkers} parallel streams...`);
      const resultBuffer = new Uint8Array(totalBytes);
      const partSize = Math.ceil(totalBytes / numWorkers);
      let loadedBytes = 0;

      const workerPromises = [];
      for (let i = 0; i < numWorkers; i++) {
        const start = i * partSize;
        const end = Math.min(start + partSize - 1, totalBytes - 1);
        if (start >= totalBytes) break;

        workerPromises.push((async (workerId, rangeStart, rangeEnd) => {
          const res = await fetch(downloadUrl, {
            headers: { "Range": `bytes=${rangeStart}-${rangeEnd}` },
            cache: "no-store"
          });
          if (!res.ok && res.status !== 206) throw new Error(`Stream worker ${workerId} failed: HTTP ${res.status}`);

          const reader = res.body.getReader();
          let writeOffset = rangeStart;
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            resultBuffer.set(value, writeOffset);
            writeOffset += value.length;
            loadedBytes += value.length;

            if (onProgress) {
              const percent = totalBytes > 0 ? Math.min(100, Math.round((loadedBytes / totalBytes) * 100)) : 0;
              onProgress(percent, loadedBytes, totalBytes);
            }
          }
        })(i, start, end));
      }

      await Promise.all(workerPromises);
      console.log(`[API] Multi-stream download complete for ${modelId}: ${(loadedBytes / (1024 * 1024)).toFixed(1)} MB`);
      return resultBuffer.buffer;
    }

    // 3. Single continuous stream for smaller models or non-range servers
    const response = await fetch(downloadUrl, { cache: "no-store" });
    if (!response.ok) {
      let errMsg = `HTTP ${response.status}: ${response.statusText}`;
      try {
        const errJson = await response.json();
        if (errJson.message) errMsg = errJson.message;
      } catch (_) {}
      throw new Error(errMsg);
    }

    const contentLength = response.headers.get("content-length");
    if (!totalBytes && contentLength) {
      totalBytes = parseInt(contentLength, 10);
    }

    if (response.body && response.body.getReader) {
      const reader = response.body.getReader();
      const chunks = [];
      let loadedBytes = 0;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        chunks.push(value);
        loadedBytes += value.length;

        if (onProgress) {
          const percent = totalBytes > 0 ? Math.min(100, Math.round((loadedBytes / totalBytes) * 100)) : 0;
          onProgress(percent, loadedBytes, totalBytes || loadedBytes);
        }
      }

      const combined = new Uint8Array(loadedBytes);
      let offset = 0;
      for (const chunk of chunks) {
        combined.set(chunk, offset);
        offset += chunk.length;
      }
      return combined.buffer;
    } else {
      const buf = await response.arrayBuffer();
      if (onProgress) onProgress(100, buf.byteLength, buf.byteLength);
      return buf;
    }
  }
}
