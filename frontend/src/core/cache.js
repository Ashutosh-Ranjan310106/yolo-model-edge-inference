/**
 * Local Model Cache using Browser IndexedDB.
 * Ensures ONNX model binaries are downloaded once from CDN or loaded via file picker,
 * stored permanently in the browser, and accessible completely offline.
 */

const DB_NAME = "ROD_Edge_Models_DB";
const STORE_NAME = "cached_models";
const DB_VERSION = 1;

export class LocalModelCache {
  constructor() {
    this.db = null;
    this.memoryStore = new Map();
  }

  async init() {
    if (this.db) return;
    return new Promise((resolve) => {
      try {
        if (typeof indexedDB === "undefined") {
          console.warn("[Cache] IndexedDB not supported; using in-memory store.");
          resolve();
          return;
        }

        const request = indexedDB.open(DB_NAME, DB_VERSION);
        const timeout = setTimeout(() => {
          console.warn("[Cache] IndexedDB open timed out after 3s; using memory fallback.");
          resolve();
        }, 3000);

        request.onerror = (e) => {
          clearTimeout(timeout);
          console.warn("[Cache] IndexedDB open error:", e.target?.error);
          resolve();
        };

        request.onsuccess = (e) => {
          clearTimeout(timeout);
          this.db = e.target.result;
          resolve();
        };

        request.onupgradeneeded = (e) => {
          const db = e.target.result;
          if (!db.objectStoreNames.contains(STORE_NAME)) {
            db.createObjectStore(STORE_NAME, { keyPath: "id" });
          }
        };

        request.onblocked = () => {
          clearTimeout(timeout);
          console.warn("[Cache] IndexedDB blocked; using memory fallback.");
          resolve();
        };
      } catch (err) {
        console.warn("[Cache] IndexedDB exception during init:", err);
        resolve();
      }
    });
  }

  async hasModel(modelKey) {
    if (this.memoryStore.has(modelKey)) return true;
    await this.init();
    if (!this.db) return false;

    return new Promise((resolve) => {
      try {
        const tx = this.db.transaction(STORE_NAME, "readonly");
        const store = tx.objectStore(STORE_NAME);
        const req = store.get(modelKey);
        req.onsuccess = () => resolve(!!req.result);
        req.onerror = () => resolve(false);
      } catch {
        resolve(false);
      }
    });
  }

  async getModel(modelKey) {
    if (this.memoryStore.has(modelKey)) {
      return this.memoryStore.get(modelKey);
    }
    await this.init();
    if (!this.db) return null;

    return new Promise((resolve) => {
      try {
        const tx = this.db.transaction(STORE_NAME, "readonly");
        const store = tx.objectStore(STORE_NAME);
        const req = store.get(modelKey);
        req.onsuccess = () => resolve(req.result || null);
        req.onerror = () => resolve(null);
      } catch {
        resolve(null);
      }
    });
  }

  async saveModel(modelKey, arrayBuffer, metadata = {}) {
    const record = {
      id: modelKey,
      buffer: arrayBuffer,
      sizeBytes: arrayBuffer.byteLength,
      savedAt: new Date().toISOString(),
      metadata: metadata
    };

    // Store in memory for immediate instant access
    this.memoryStore.set(modelKey, record);

    await this.init();
    if (!this.db) {
      return record;
    }

    return new Promise((resolve) => {
      let settled = false;
      const safeResolve = () => {
        if (!settled) {
          settled = true;
          resolve(record);
        }
      };

      const timer = setTimeout(() => {
        console.warn(`[Cache] IndexedDB save for ${modelKey} timed out; using memory.`);
        safeResolve();
      }, 4000);

      try {
        const tx = this.db.transaction(STORE_NAME, "readwrite");
        tx.oncomplete = () => {
          clearTimeout(timer);
          safeResolve();
        };
        tx.onerror = () => {
          clearTimeout(timer);
          safeResolve();
        };
        tx.onabort = () => {
          clearTimeout(timer);
          safeResolve();
        };
        const store = tx.objectStore(STORE_NAME);
        store.put(record);
      } catch (err) {
        clearTimeout(timer);
        safeResolve();
      }
    });
  }

  async deleteModel(modelKey) {
    this.memoryStore.delete(modelKey);
    await this.init();
    if (!this.db) return;

    return new Promise((resolve) => {
      try {
        const tx = this.db.transaction(STORE_NAME, "readwrite");
        const store = tx.objectStore(STORE_NAME);
        store.delete(modelKey);
        tx.oncomplete = () => resolve();
        tx.onerror = () => resolve();
      } catch {
        resolve();
      }
    });
  }

  async clearAll() {
    this.memoryStore.clear();
    await this.init();
    if (!this.db) return;

    return new Promise((resolve) => {
      try {
        const tx = this.db.transaction(STORE_NAME, "readwrite");
        const store = tx.objectStore(STORE_NAME);
        store.clear();
        tx.oncomplete = () => resolve();
        tx.onerror = () => resolve();
      } catch {
        resolve();
      }
    });
  }

  async listCachedModels() {
    await this.init();
    if (!this.db) {
      return Array.from(this.memoryStore.values()).map(r => ({
        id: r.id,
        sizeBytes: r.sizeBytes,
        savedAt: r.savedAt
      }));
    }

    return new Promise((resolve, reject) => {
      const tx = this.db.transaction(STORE_NAME, "readonly");
      const store = tx.objectStore(STORE_NAME);
      const req = store.getAll();
      req.onsuccess = () => {
        const list = (req.result || []).map(item => ({
          id: item.id,
          sizeBytes: item.sizeBytes,
          savedAt: item.savedAt
        }));
        resolve(list);
      };
      req.onerror = (e) => reject(e.target.error);
    });
  }
}

export const modelCache = new LocalModelCache();
