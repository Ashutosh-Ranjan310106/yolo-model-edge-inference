/**
 * Local Model Cache using Browser IndexedDB.
 * Ensures models are downloaded once, validated, and loaded offline.
 */
const DB_NAME = "ROD_Edge_Models_DB";
const STORE_NAME = "cached_models";
const DB_VERSION = 1;

export class LocalModelCache {
  constructor() {
    this.db = null;
    this.memoryStore = new Map(); // In-memory fallback if IndexedDB is unavailable or quota exceeded
  }

  async init() {
    if (this.db) return;
    return new Promise((resolve) => {
      try {
        if (typeof indexedDB === "undefined") {
          console.warn("[Cache] IndexedDB not supported in this environment. Using in-memory fallback.");
          resolve();
          return;
        }

        const request = indexedDB.open(DB_NAME, DB_VERSION);
        
        // Don't hang indefinitely if blocked
        const timeout = setTimeout(() => {
          console.warn("[Cache] IndexedDB open timed out after 3s. Proceeding with memory fallback.");
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
          console.warn("[Cache] IndexedDB open blocked. Proceeding with memory fallback.");
          resolve();
        };
      } catch (err) {
        console.warn("[Cache] IndexedDB exception during init:", err);
        resolve();
      }
    });
  }

  async hasModel(modelId) {
    if (this.memoryStore.has(modelId)) return true;
    await this.init();
    if (!this.db) return false;

    return new Promise((resolve) => {
      try {
        const tx = this.db.transaction(STORE_NAME, "readonly");
        const store = tx.objectStore(STORE_NAME);
        const req = store.get(modelId);
        req.onsuccess = () => resolve(!!req.result);
        req.onerror = () => resolve(false);
      } catch {
        resolve(false);
      }
    });
  }

  async getModel(modelId) {
    if (this.memoryStore.has(modelId)) {
      return this.memoryStore.get(modelId);
    }
    await this.init();
    if (!this.db) return null;

    return new Promise((resolve) => {
      try {
        const tx = this.db.transaction(STORE_NAME, "readonly");
        const store = tx.objectStore(STORE_NAME);
        const req = store.get(modelId);
        req.onsuccess = () => resolve(req.result || null);
        req.onerror = () => resolve(null);
      } catch {
        resolve(null);
      }
    });
  }

  async saveModel(modelId, arrayBuffer, metadata = {}) {
    const record = {
      id: modelId,
      buffer: arrayBuffer,
      sizeBytes: arrayBuffer.byteLength,
      savedAt: new Date().toISOString(),
      version: metadata.version || "1.0",
      sha256: metadata.sha256 || "",
      metadata: metadata
    };

    // Always store in memory for immediate instant access
    this.memoryStore.set(modelId, record);

    await this.init();
    if (!this.db) {
      console.log(`[Cache] Model ${modelId} saved to memory cache.`);
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

      // 4-second safety timeout so saving to IndexedDB NEVER freezes the UI
      const timer = setTimeout(() => {
        console.warn(`[Cache] IndexedDB put for ${modelId} took > 4s, falling back to memory.`);
        safeResolve();
      }, 4000);

      try {
        const tx = this.db.transaction(STORE_NAME, "readwrite");
        
        tx.oncomplete = () => {
          clearTimeout(timer);
          safeResolve();
        };

        tx.onerror = (e) => {
          clearTimeout(timer);
          console.warn(`[Cache] Transaction error saving ${modelId}:`, e.target?.error);
          safeResolve(); // Already stored in memoryStore!
        };

        tx.onabort = (e) => {
          clearTimeout(timer);
          console.warn(`[Cache] Transaction aborted saving ${modelId}:`, e.target?.error);
          safeResolve(); // Already stored in memoryStore!
        };

        const store = tx.objectStore(STORE_NAME);
        store.put(record);
      } catch (err) {
        clearTimeout(timer);
        console.warn(`[Cache] Exception putting ${modelId} into store:`, err);
        safeResolve();
      }
    });
  }

  async deleteModel(modelId) {
    this.memoryStore.delete(modelId);
    await this.init();
    if (!this.db) return;

    return new Promise((resolve) => {
      try {
        const tx = this.db.transaction(STORE_NAME, "readwrite");
        const store = tx.objectStore(STORE_NAME);
        store.delete(modelId);
        tx.oncomplete = () => resolve();
        tx.onerror = () => resolve();
      } catch {
        resolve();
      }
    });
  }

  async listCachedModels() {
    await this.init();
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction(STORE_NAME, "readonly");
      const store = tx.objectStore(STORE_NAME);
      const req = store.getAll();
      req.onsuccess = () => {
        const list = (req.result || []).map(item => ({
          id: item.id,
          sizeBytes: item.sizeBytes,
          savedAt: item.savedAt,
          version: item.version,
          sha256: item.sha256
        }));
        resolve(list);
      };
      req.onerror = (e) => reject(e.target.error);
    });
  }
}

export const modelCache = new LocalModelCache();
