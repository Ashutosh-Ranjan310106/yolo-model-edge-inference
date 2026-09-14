#!/usr/bin/env python3
"""
ROD Model Preparation Script.

Exports PyTorch checkpoints (.pt) to mobile-optimized formats (ONNX, NCNN)
BEFORE inference or distribution.

Steps:
1. Find the original .pt checkpoint.
2. Export it to the requested mobile format and resolution.
3. Save to weights/mobile/.
4. Validate the exported model (e.g. onnx.checker).
5. Record file size.
6. Calculate SHA256 checksum.
7. Register the model variant in model_registry.json.

Usage:
  python scripts/prepare_model.py --model yolo26s-nav-291_best.pt --resolution 480 --format onnx
  python scripts/prepare_model.py --model runs/train/yolo26s-nav-291/weights/best.pt --resolution 480 --format onnx
  python scripts/prepare_model.py --scan-existing  # Registers already exported mobile models
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

# Setup paths relative to backend root
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
WEIGHTS_DIR = BACKEND_DIR / "weights"
MOBILE_DIR = WEIGHTS_DIR / "mobile"
ORIGINAL_DIR = WEIGHTS_DIR / "original"
REGISTRY_FILE = BACKEND_DIR / "model_registry.json"
NAV_DIR = BACKEND_DIR.parent.parent
RUNS_DIR = NAV_DIR / "runs" / "train"

MOBILE_DIR.mkdir(parents=True, exist_ok=True)
ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_ROD_CLASSES = [
    "Bike", "Building", "Car", "Person", "Stairs",
    "Traffic sign", "Electrical Pole", "Road", "Motorcycle", "Dustbin",
    "Dog", "Manhole", "Tree", "Guard rail", "Pedestrian crosswalk",
    "Truck", "Bus", "Bench", "Traffic Cone", "Fire hydrant",
    "Teraffic Barrel", "Plant Pot", "Electrical Box", "Chair",
    "Bicycle Rack", "Door", "Wall"
]


def calculate_sha256(file_path: Path) -> str:
    """Calculate SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_registry() -> Dict[str, Any]:
    """Load model_registry.json or initialize if missing."""
    if REGISTRY_FILE.exists():
        try:
            with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[!] Warning: Could not read {REGISTRY_FILE.name}: {e}. Reinitializing.")
    return {"version": "1.0", "updated_at": "", "models": {}}


def save_registry(registry_data: Dict[str, Any]):
    """Save model_registry.json atomically."""
    import datetime
    registry_data["updated_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    temp_file = REGISTRY_FILE.with_suffix(".tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(registry_data, f, indent=2)
    temp_file.replace(REGISTRY_FILE)
    print(f"[OK] Saved model registry to {REGISTRY_FILE.name}")


def find_original_model(model_ref: str) -> Optional[Path]:
    """
    Locates original .pt model from:
    1. Exact path
    2. weights/original/
    3. runs/train/<run>/weights/
    4. navigation/ root (.pt)
    """
    p = Path(model_ref)
    if p.exists() and p.is_file():
        return p.resolve()

    # Check in weights/original/
    in_orig = ORIGINAL_DIR / model_ref
    if in_orig.exists():
        return in_orig.resolve()
    if not model_ref.endswith(".pt"):
        in_orig_pt = ORIGINAL_DIR / f"{model_ref}.pt"
        if in_orig_pt.exists():
            return in_orig_pt.resolve()

    # Check in runs/train/
    if RUNS_DIR.exists():
        # Maybe model_ref is run_id (e.g. yolo26s-nav-291)
        run_best = RUNS_DIR / model_ref / "weights" / "best.pt"
        if run_best.exists():
            return run_best.resolve()
        
        # Maybe model_ref has _best or _last suffix
        clean_name = model_ref.replace(".pt", "")
        if clean_name.endswith("_best"):
            r_id = clean_name[:-5]
            cand = RUNS_DIR / r_id / "weights" / "best.pt"
            if cand.exists():
                return cand.resolve()
        elif clean_name.endswith("_last"):
            r_id = clean_name[:-5]
            cand = RUNS_DIR / r_id / "weights" / "last.pt"
            if cand.exists():
                return cand.resolve()

        # Search recursively in runs/train/
        for candidate in RUNS_DIR.glob(f"**/{model_ref}"):
            if candidate.is_file():
                return candidate.resolve()
        if not model_ref.endswith(".pt"):
            for candidate in RUNS_DIR.glob(f"**/{model_ref}.pt"):
                if candidate.is_file():
                    return candidate.resolve()

    # Check navigation root
    for pt in NAV_DIR.glob("*.pt"):
        if pt.name == model_ref or pt.stem == model_ref:
            return pt.resolve()

    return None


def validate_onnx_model(onnx_path: Path) -> bool:
    """Validate ONNX model structure using onnx.checker."""
    try:
        import onnx
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        print(f"[OK] ONNX checker validated: {onnx_path.name}")
        return True
    except ImportError:
        print("[!] onnx python package not installed, skipping onnx.checker")
        return onnx_path.exists() and onnx_path.stat().st_size > 0
    except Exception as e:
        print(f"[ERROR] ONNX validation failed for {onnx_path.name}: {e}")
        return False


def register_variant(
    model_id: str,
    name: str,
    resolution: int,
    format_type: str,
    file_path: Path,
    source_pt: Optional[Path] = None,
    category: str = "training_best",
    variant: str = "n",
    classes: Optional[List[str]] = None
):
    """Register or update a model variant in model_registry.json."""
    registry = load_registry()
    models = registry.setdefault("models", {})

    if model_id not in models:
        models[model_id] = {
            "id": model_id,
            "name": name,
            "category": category,
            "variant": variant,
            "source_checkpoint": str(source_pt) if source_pt else "",
            "classes": len(classes or DEFAULT_ROD_CLASSES),
            "class_names": classes or DEFAULT_ROD_CLASSES,
            "variants": []
        }

    # Relative path from backend root for portability
    rel_path = f"weights/mobile/{file_path.name}"
    size_bytes = file_path.stat().st_size
    sha256 = calculate_sha256(file_path)

    # Check if this resolution & format variant already recorded
    existing_variants = models[model_id]["variants"]
    found = False
    for v in existing_variants:
        if v.get("resolution") == resolution and v.get("format") == format_type:
            v["path"] = rel_path
            v["size_bytes"] = size_bytes
            v["sha256"] = sha256
            v["status"] = "ready"
            found = True
            break

    if not found:
        existing_variants.append({
            "resolution": resolution,
            "format": format_type,
            "path": rel_path,
            "size_bytes": size_bytes,
            "sha256": sha256,
            "status": "ready"
        })

    save_registry(registry)
    print(f"[OK] Registered '{model_id}' (res={resolution}, format={format_type}) in registry.")


def prepare_model(
    model_ref: str,
    resolution: int = 480,
    format_type: str = "onnx",
    opset: int = 20,
    simplify: bool = True
) -> Path:
    """
    Executes full preparation pipeline:
    .pt -> export -> weights/mobile/ -> validate -> SHA256 -> register
    """
    print(f"\n=======================================================")
    print(f" ROD MODEL PREPARATION PIPELINE")
    print(f" Model:      {model_ref}")
    print(f" Resolution: {resolution}x{resolution}")
    print(f" Format:     {format_type}")
    print(f"=======================================================\n")

    # 1. Locate original .pt model
    source_pt = find_original_model(model_ref)
    if not source_pt:
        raise FileNotFoundError(f"Could not locate original .pt model for '{model_ref}'")

    print(f"[1/6] Found original checkpoint: {source_pt}")

    # Deduce clean model_id
    parent_dir = source_pt.parent.parent.name  # run folder if in runs/train/<run>/weights/
    stem = source_pt.stem
    if "runs" in str(source_pt) and parent_dir:
        model_id = f"{parent_dir}_{stem}"
    else:
        model_id = stem

    variant_char = "n"
    for char in ["s", "m", "l", "x"]:
        if f"26{char}" in model_id.lower() or f"v8{char}" in model_id.lower():
            variant_char = char
            break

    target_filename = f"{model_id}_{resolution}.{format_type}"
    target_path = MOBILE_DIR / target_filename

    # 2. Export using ultralytics
    print(f"[2/6] Loading checkpoint into YOLO...")
    from ultralytics import YOLO
    model = YOLO(str(source_pt))

    print(f"[3/6] Exporting to {format_type.upper()} (imgsz={resolution}, opset={opset}, simplify={simplify})...")
    if format_type.lower() == "onnx":
        exported_path = model.export(
            format="onnx",
            imgsz=resolution,
            opset=opset,
            simplify=simplify
        )
        if not exported_path or not Path(exported_path).exists():
            raise RuntimeError("YOLO export completed but output file not found.")

        # 4. Save to weights/mobile/
        shutil.move(exported_path, target_path)
        print(f"[4/6] Saved mobile model to {target_path}")

        # 5. Validate model
        print(f"[5/6] Validating ONNX model integrity...")
        if not validate_onnx_model(target_path):
            raise RuntimeError(f"Validation failed for exported model {target_path}")

    elif format_type.lower() == "ncnn":
        exported_path = model.export(format="ncnn", imgsz=resolution)
        if not exported_path or not Path(exported_path).exists():
            raise RuntimeError("NCNN export failed.")
        target_dir = MOBILE_DIR / f"{model_id}_{resolution}_ncnn"
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.move(exported_path, target_dir)
        target_path = target_dir
        print(f"[4/6] Saved NCNN directory to {target_path}")
    else:
        raise ValueError(f"Unsupported mobile format: {format_type}")

    # 6. Record file size, SHA256 & Register
    print(f"[6/6] Computing SHA256 and registering model...")
    class_names = [model.names[i] for i in sorted(model.names.keys())] if hasattr(model, "names") else DEFAULT_ROD_CLASSES

    register_variant(
        model_id=model_id,
        name=f"{model_id} ({source_pt.name})",
        resolution=resolution,
        format_type=format_type,
        file_path=target_path,
        source_pt=source_pt,
        category="training_best" if "best" in stem else "training_checkpoint",
        variant=variant_char,
        classes=class_names
    )

    print(f"\n[OK] Model preparation complete: {target_path.name}")
    print(f"    Size: {target_path.stat().st_size / (1024 * 1024):.2f} MB")
    print(f"    SHA256: {calculate_sha256(target_path)}")
    return target_path


def scan_and_register_existing():
    """Scans weights/mobile/ and registers all existing prepared models."""
    print("\nScanning existing models in weights/mobile/...")
    count = 0
    for f in sorted(MOBILE_DIR.glob("*.onnx")):
        # e.g. yolo26s-nav-291_best_480.onnx -> model_id: yolo26s-nav-291_best, res: 480
        parts = f.stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            model_id = parts[0]
            res = int(parts[1])
        else:
            model_id = f.stem
            res = 480

        var_char = "n"
        for c in ["s", "m", "l", "x"]:
            if f"26{c}" in model_id.lower():
                var_char = c
                break

        # Validate with ONNX
        validate_onnx_model(f)

        register_variant(
            model_id=model_id,
            name=f"{model_id} (ONNX - {res}p)",
            resolution=res,
            format_type="onnx",
            file_path=f,
            category="training_best" if "best" in model_id else "base_weight",
            variant=var_char,
            classes=DEFAULT_ROD_CLASSES
        )
        count += 1

    print(f"[OK] Scanned and registered {count} existing mobile ONNX models.")


def main():
    parser = argparse.ArgumentParser(description="Prepare and register mobile YOLO models.")
    parser.add_argument("--model", type=str, help="Model ID, path, or filename (e.g. yolo26s-nav-291_best.pt)")
    parser.add_argument("--resolution", type=int, default=480, choices=[320, 384, 480, 512, 640], help="Target square resolution")
    parser.add_argument("--format", type=str, default="onnx", choices=["onnx", "ncnn"], help="Mobile format")
    parser.add_argument("--opset", type=int, default=20, help="ONNX opset version")
    parser.add_argument("--no-simplify", action="store_true", help="Disable onnx-simplifier")
    parser.add_argument("--scan-existing", action="store_true", help="Scan and register already exported models in weights/mobile/")
    args = parser.parse_args()

    if args.scan_existing:
        scan_and_register_existing()
        return

    if not args.model:
        parser.print_help()
        sys.exit(1)

    try:
        prepare_model(
            model_ref=args.model,
            resolution=args.resolution,
            format_type=args.format,
            opset=args.opset,
            simplify=not args.no_simplify
        )
    except Exception as e:
        print(f"\n[ERROR] during preparation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
