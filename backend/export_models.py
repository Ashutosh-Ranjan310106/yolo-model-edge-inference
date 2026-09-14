"""
ROD Model Export Pipeline for Edge Inference.
Exports PyTorch YOLO models (.pt) into optimized mobile runtime formats:
- ONNX (optimized via onnxslim for ONNX Runtime Web / Android)
- NCNN (model.ncnn.param + model.ncnn.bin for Android NCNN)
"""

import sys
import shutil
import argparse
from pathlib import Path
from ultralytics import YOLO

CURRENT_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = CURRENT_DIR / "weights"

def export_model(pt_path: Path, output_dir: Path, resolutions: list[int], formats: list[str]):
    """Export a trained YOLO .pt model to mobile formats at specified resolutions."""
    if not pt_path.exists():
        raise FileNotFoundError(f"PyTorch weights not found at: {pt_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n========================================================")
    print(f"Exporting: {pt_path.name}")
    print(f"Source   : {pt_path}")
    print(f"Target   : {output_dir}")
    print(f"========================================================")

    # Determine base model name (e.g., 'yolov26n' or 'yolov26s')
    stem = pt_path.stem
    parent_name = pt_path.parent.parent.name
    if "26n" in stem.lower() or "26n" in parent_name.lower():
        base_name = "yolov26n"
    elif "26s" in stem.lower() or "26s" in parent_name.lower():
        base_name = "yolov26s"
    elif "26m" in stem.lower() or "26m" in parent_name.lower():
        base_name = "yolov26m"
    else:
        base_name = "yolo_model"

    # Also keep a copy of the .pt weights in weights_dir
    stored_pt = output_dir / f"{base_name}_best.pt"
    if not stored_pt.exists() or stored_pt.stat().st_size != pt_path.stat().st_size:
        print(f"[*] Copying PyTorch checkpoint to: {stored_pt.name}")
        shutil.copy2(pt_path, stored_pt)

    # Load model once
    model = YOLO(str(stored_pt))
    print(f"[OK] Loaded YOLO model with {len(model.names)} classes: {list(model.names.values())[:5]}...")

    for res in resolutions:
        print(f"\n--- Processing Resolution: {res}x{res} ---")

        # 1. Export ONNX format
        if "onnx" in formats:
            onnx_target_name = f"{base_name}_{res}.onnx"
            onnx_target_path = output_dir / onnx_target_name
            if onnx_target_path.exists():
                print(f"[SKIP] ONNX model already exists: {onnx_target_name} ({onnx_target_path.stat().st_size / 1e6:.2f} MB)")
            else:
                print(f"[*] Exporting ONNX ({res}x{res})...")
                try:
                    exported = model.export(format="onnx", imgsz=res, opset=20, simplify=True)
                    if exported and Path(exported).exists():
                        shutil.move(exported, onnx_target_path)
                        print(f"[SUCCESS] ONNX saved as: {onnx_target_name} ({onnx_target_path.stat().st_size / 1e6:.2f} MB)")
                except Exception as e:
                    print(f"[ERROR] ONNX export failed for {res}: {e}")

        # 2. Export NCNN format
        if "ncnn" in formats:
            ncnn_target_dir = output_dir / f"{base_name}_{res}_ncnn"
            if ncnn_target_dir.exists() and (ncnn_target_dir / "model.ncnn.param").exists():
                print(f"[SKIP] NCNN model already exists: {ncnn_target_dir.name}")
            else:
                print(f"[*] Exporting NCNN ({res}x{res})...")
                try:
                    exported = model.export(format="ncnn", imgsz=res)
                    if exported and Path(exported).exists():
                        if ncnn_target_dir.exists():
                            shutil.rmtree(ncnn_target_dir)
                        shutil.move(exported, ncnn_target_dir)
                        print(f"[SUCCESS] NCNN saved to: {ncnn_target_dir.name}")
                except Exception as e:
                    print(f"[ERROR] NCNN export failed for {res}: {e}")

    print(f"\n[DONE] All exports completed in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Export trained YOLO models for edge inference.")
    parser.add_argument(
        "--weights",
        type=str,
        default=r"c:\Users\rrpra\Documents\Github\sih\DRISHTIX\navigation\runs\train\yolo26n-24k-nav-run1\weights\best.pt",
        help="Path to trained .pt file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(WEIGHTS_DIR),
        help="Target output directory"
    )
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=[480, 320, 640],
        help="Input resolutions to export (e.g. 480 320 640)"
    )
    parser.add_argument(
        "--formats",
        type=str,
        nargs="+",
        default=["onnx", "ncnn"],
        help="Formats to export (onnx, ncnn)"
    )

    args = parser.parse_args()
    pt_path = Path(args.weights)
    output_dir = Path(args.output)

    export_model(pt_path, output_dir, args.resolutions, args.formats)

if __name__ == "__main__":
    main()
