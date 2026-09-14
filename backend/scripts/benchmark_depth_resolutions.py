"""
Benchmark script comparing YOLO26 Depth inference performance at 512x512 vs 320x320.
Measures latency, throughput (FPS), tensor dimensions, and ROI extraction overhead.
"""

import os
import sys
import time
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("onnxruntime not installed in this environment.")
    sys.exit(1)

DEPTH_MODELS = {
    "YOLO26-Depth @ 512x512 (Quality / Default)": {
        "res": 512,
        "path": os.path.join(os.path.dirname(__file__), "..", "..", "models", "depth", "depth_512.onnx")
    },
    "YOLO26-Depth @ 320x320 (Speed / Low-Power)": {
        "res": 320,
        "path": os.path.join(os.path.dirname(__file__), "..", "..", "models", "depth", "depth_320.onnx")
    }
}

def benchmark_model(name, config, n_warmup=3, n_runs=10):
    model_path = os.path.abspath(config["path"])
    res = config["res"]

    if not os.path.exists(model_path):
        return {
            "name": name,
            "error": f"Model file not found: {model_path}"
        }

    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    # Session options mimicking mobile single-thread WASM / mobile CPU
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 2
    sess_opts.inter_op_num_threads = 1
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    t_load_start = time.perf_counter()
    session = ort.InferenceSession(model_path, sess_opts, providers=["CPUExecutionProvider"])
    load_time_ms = (time.perf_counter() - t_load_start) * 1000

    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_name = session.get_outputs()[0].name
    output_shape = session.get_outputs()[0].shape

    dummy_input = np.random.randn(1, 3, res, res).astype(np.float32)

    # Warmup
    for _ in range(n_warmup):
        session.run([output_name], {input_name: dummy_input})

    # Benchmark
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out = session.run([output_name], {input_name: dummy_input})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    avg_lat = float(np.mean(latencies))
    min_lat = float(np.min(latencies))
    max_lat = float(np.max(latencies))
    std_lat = float(np.std(latencies))
    fps = 1000.0 / avg_lat

    # Benchmark ROI extraction overhead
    depth_map = out[0][0, 0] # [res, res]
    t_roi_start = time.perf_counter()
    for _ in range(100):
        # Sample an ROI (e.g. 20% to 60% coordinates)
        x1, y1, x2, y2 = int(res * 0.2), int(res * 0.2), int(res * 0.6), int(res * 0.6)
        sub = depth_map[y1:y2:2, x1:x2:2]
        med = float(np.median(sub))
    roi_lat_us = ((time.perf_counter() - t_roi_start) / 100) * 1000000

    return {
        "name": name,
        "res": res,
        "file_size_mb": file_size_mb,
        "load_time_ms": load_time_ms,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "avg_latency_ms": avg_lat,
        "min_latency_ms": min_lat,
        "max_latency_ms": max_lat,
        "std_latency_ms": std_lat,
        "fps": fps,
        "roi_overhead_us": roi_lat_us
    }

def main():
    print("=" * 80)
    print("  YOLO26-Depth Model Benchmark: 512x512 (Quality) vs 320x320 (Speed)")
    print("=" * 80)
    print()

    results = []
    for name, config in DEPTH_MODELS.items():
        print(f"Running benchmark for {name}...")
        res = benchmark_model(name, config)
        results.append(res)

    print()
    print("-" * 80)
    print(f"{'Configuration':<42} | {'Input Res':<10} | {'Latency':<10} | {'FPS':<8} | {'Load Time'}")
    print("-" * 80)

    for r in results:
        if "error" in r:
            print(f"{r['name']:<42} | ERROR: {r['error']}")
        else:
            print(f"{r['name']:<42} | {r['res']}x{r['res']:<6} | {r['avg_latency_ms']:.1f} ms    | {r['fps']:.1f}    | {r['load_time_ms']:.0f} ms")

    print("-" * 80)
    print()

    if len(results) == 2 and "avg_latency_ms" in results[0] and "avg_latency_ms" in results[1]:
        speedup = results[0]["avg_latency_ms"] / results[1]["avg_latency_ms"]
        print(f"-> 320x320 is {speedup:.2f}x faster than 512x512.")
        print(f"-> Latency drop: from {results[0]['avg_latency_ms']:.1f}ms down to {results[1]['avg_latency_ms']:.1f}ms.")
    print()

if __name__ == "__main__":
    main()
