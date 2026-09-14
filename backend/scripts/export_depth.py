import os
import sys
import shutil
import time
from pathlib import Path
import onnx
from ultralytics import YOLO

def export_depth_models():
    base_dir = Path(__file__).resolve().parent.parent.parent
    pt_path = base_dir / "yolo26n-depth.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"Source PyTorch checkpoint not found: {pt_path}")

    # Destination directories: models/depth/
    dest_dir = base_dir / "models" / "depth"
    dest_dir.mkdir(parents=True, exist_ok=True)

    dist_dir = base_dir.parent / "mobile_distributable_models"
    dist_dir.mkdir(parents=True, exist_ok=True)

    # Also sync to backend/weights/depth/
    weights_depth_dir = base_dir / "backend" / "weights" / "depth"
    weights_depth_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading pretrained weights from {pt_path}...")
    model = YOLO(str(pt_path))

    resolutions = [512, 320]
    exported_files = {}

    for res in resolutions:
        print(f"\n==================================================")
        print(f"Exporting YOLO26-Depth at {res}x{res}...")
        print(f"==================================================")
        t0 = time.time()
        
        # Proper Ultralytics export
        out_onnx_name = model.export(format="onnx", imgsz=res, simplify=True)
        export_time = time.time() - t0
        print(f"Export completed in {export_time:.2f}s -> {out_onnx_name}")

        temp_onnx = Path(out_onnx_name)
        target_name = f"depth_{res}.onnx"
        target_path = dest_dir / target_name

        # Copy to models/depth/
        shutil.copy2(temp_onnx, target_path)
        print(f"Saved to {target_path} ({target_path.stat().st_size / (1024*1024):.2f} MB)")

        # Copy to mobile_distributable_models/
        shutil.copy2(target_path, dist_dir / target_name)
        print(f"Copied to {dist_dir / target_name}")

        # Copy to backend/weights/depth/
        shutil.copy2(target_path, weights_depth_dir / target_name)

        # Verify ONNX model integrity
        onnx_model = onnx.load(str(target_path))
        onnx.checker.check_model(onnx_model)
        
        input_shape = [d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]
        output_shape = [d.dim_value for d in onnx_model.graph.output[0].type.tensor_type.shape.dim]

        print(f"VERIFIED: Input Shape={input_shape}, Output Shape={output_shape}")
        exported_files[res] = {
            "path": str(target_path),
            "size_mb": round(target_path.stat().st_size / (1024*1024), 2),
            "input_shape": input_shape,
            "output_shape": output_shape,
            "export_time_s": round(export_time, 2)
        }

    print("\nAll depth models exported and verified successfully:")
    for res, info in exported_files.items():
        print(f" - {res}x{res}: {info['path']} ({info['size_mb']} MB, Input: {info['input_shape']}, Output: {info['output_shape']})")

if __name__ == "__main__":
    export_depth_models()
