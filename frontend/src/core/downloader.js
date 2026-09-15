/**
 * Model Downloader & Local File Loader.
 * Downloads models from public CDN mirrors with progress tracking,
 * or reads directly from user-selected local .onnx files with zero internet.
 */

export async function downloadFromUrl(url, onProgress = null) {
  const timestamp = Date.now();
  const fetchUrl = url.includes("?") ? `${url}&_t=${timestamp}` : `${url}?_t=${timestamp}`;

  const response = await fetch(fetchUrl);
  if (!response.ok) {
    throw new Error(`Failed to download model from ${url}: HTTP ${response.status}`);
  }

  const contentLength = response.headers.get("content-length");
  const totalBytes = contentLength ? parseInt(contentLength, 10) : 0;

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
        onProgress(percent, loadedBytes, totalBytes);
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

export async function readFileAsArrayBuffer(file, onProgress = null) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onprogress = (e) => {
      if (e.lengthComputable && onProgress) {
        const percent = Math.round((e.loaded / e.total) * 100);
        onProgress(percent, e.loaded, e.total);
      }
    };

    reader.onload = () => {
      if (onProgress) onProgress(100, file.size, file.size);
      resolve(reader.result);
    };

    reader.onerror = () => reject(new Error("Failed to read local file"));
    reader.readAsArrayBuffer(file);
  });
}
