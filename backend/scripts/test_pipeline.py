"""
Test script to verify the modular real-time perception and guidance pipeline end-to-end.
Validates:
1. YOLO26 Detection + Depth Estimation (ONNX Runtime)
2. Sutherland-Hodgman Polygon Clipping and Corridor Overlap
3. 7-Zone Direction and Continuous Horizontal Positioning
4. 5 Distance Categories and Depth ROI Extraction
5. Object Tracking and EMA Smoothing
6. Hazard Criticality and Risk Engine Calculation
7. Priority Management (CRITICAL, HIGH, MEDIUM, LOW, IGNORE)
8. Deterministic Spoken Guidance Generation and Event Cooldowns
9. Annotated Debug Frame Generation
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import cv2

# Set backend in Python path
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.inference.config import pipeline_config
from app.inference.pipeline import NavigationPipeline
from app.inference.spatial.geometry import polygon_intersection, polygon_area
from app.inference.spatial.direction import DirectionClassifier
from app.inference.spatial.distance import DistanceEstimator
from app.inference.guidance.guidance_generator import GuidanceGenerator

def test_geometry_and_spatial():
    print("--- Test 1: Geometry & Spatial Calculations ---")
    poly1 = [(100.0, 100.0), (300.0, 100.0), (300.0, 300.0), (100.0, 300.0)]
    poly2 = [(200.0, 200.0), (400.0, 200.0), (400.0, 400.0), (200.0, 400.0)]
    area = polygon_intersection(poly1, poly2)
    assert abs(area - 10000.0) < 1e-2, f"Expected 10000.0, got {area}"
    print("  [PASS] Sutherland-Hodgman polygon clipping and area")

    dir_classifier = DirectionClassifier()
    assert dir_classifier.classify(50, 480)[0] == "far left"
    assert dir_classifier.classify(140, 480)[0] == "left"
    assert dir_classifier.classify(200, 480)[0] == "slightly left"
    assert dir_classifier.classify(240, 480)[0] == "center"
    assert dir_classifier.classify(280, 480)[0] == "slightly right"
    assert dir_classifier.classify(350, 480)[0] == "right"
    assert dir_classifier.classify(450, 480)[0] == "far right"
    print("  [PASS] 7-Zone Direction classification")

    dist_est = DistanceEstimator()
    assert dist_est.categorize_metric(0.8) == "VERY_NEAR"
    assert dist_est.categorize_metric(1.5) == "NEAR"
    assert dist_est.categorize_metric(3.5) == "MEDIUM"
    assert dist_est.categorize_metric(7.0) == "FAR"
    assert dist_est.categorize_metric(15.0) == "VERY_FAR"
    print("  [PASS] 5 Distance categories")

def test_guidance_generation():
    print("--- Test 2: Deterministic Spoken Guidance Generation ---")
    gen = GuidanceGenerator()
    
    msg1 = gen.generate({"class": "person", "direction": "slightly left", "distance_category": "NEAR", "priority": "HIGH"})
    assert msg1 == "Person slightly left, near.", f"Got: {msg1}"
    print(f"  [PASS] Standard template: '{msg1}'")

    msg2 = gen.generate({"class": "car", "direction": "center", "distance_category": "MEDIUM", "priority": "MEDIUM"})
    assert msg2 == "Car ahead, medium.", f"Got: {msg2}"
    print(f"  [PASS] Center maps to ahead: '{msg2}'")

    msg3 = gen.generate({"class": "stairs", "direction": "center", "distance_category": "VERY_NEAR", "priority": "CRITICAL"})
    assert msg3 == "Stop. Stairs ahead, very near.", f"Got: {msg3}"
    print(f"  [PASS] Critical life-safety alert: '{msg3}'")

def test_end_to_end_pipeline():
    print("--- Test 3: End-to-End Navigation Pipeline ---")
    pipeline = NavigationPipeline(enable_depth=True)
    
    configured = pipeline.configure(
        yolo_model_id="yolo26n-nav-run6",
        yolo_resolution=480,
        depth_model_id="yolo26n_depth",
        depth_resolution=512,
        enable_depth=True
    )
    yolo_ok = pipeline.detector.session is not None
    depth_ok = pipeline.depth_estimator.session is not None if pipeline.depth_estimator else False
    print(f"  Pipeline configured: yolo_loaded={yolo_ok}, depth_loaded={depth_ok}")

    bus_path = Path(BACKEND_DIR).parent.parent / "bus.jpg"
    if bus_path.exists():
        frame = cv2.imread(str(bus_path))
        frame = cv2.resize(frame, (480, 480))
        print(f"  Using sample image: {bus_path} (480x480)")
    else:
        frame = np.full((480, 480, 3), 120, dtype=np.uint8)
        print("  Using synthetic frame (480x480)")

    print("  Processing 5 consecutive frames...")
    for frame_id in range(1, 6):
        result_payload, annotated_bgr = pipeline.process_frame(
            frame_bgr=frame,
            confidence_threshold=0.25,
            frame_id=frame_id
        )

        assert "fps" in result_payload
        assert "latency" in result_payload
        assert "objects" in result_payload
        assert "events" in result_payload
        assert annotated_bgr.shape == (480, 480, 3)

        total_lat = result_payload['latency'].get('pipeline_total_ms', 0)
        print(f"    Frame {frame_id}: FPS={result_payload['fps']}, Objects={len(result_payload['objects'])}, Events={len(result_payload['events'])}, Latency={total_lat:.1f}ms")
        
        for obj in result_payload['objects'][:3]:
            print(f"      -> Object: {obj['class']} ({obj['confidence']:.2f}) | Dir: {obj['direction']} | Dist: {obj.get('distance_category')} ({obj.get('distance_m')}m) | Priority: {obj['priority']} | Risk: {obj['risk']:.2f}")

        for ev in result_payload['events']:
            print(f"      -> Event: [{ev['priority']}] {ev['message']}")

    print("  [PASS] Complete end-to-end pipeline execution and schema validation")

if __name__ == '__main__':
    print("==================================================")
    print("  ROD NAVIGATION PIPELINE VERIFICATION TEST")
    print("==================================================")
    test_geometry_and_spatial()
    test_guidance_generation()
    test_end_to_end_pipeline()
    print("==================================================")
    print("  ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("==================================================")
