import base64
import json
import time
import struct
from typing import Optional
import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.inference.pipeline import navigation_pipeline
from app.services.session_recorder import session_recorder

router = APIRouter(tags=["websocket"])

@router.websocket("/ws/inference")
async def websocket_inference_endpoint(websocket: WebSocket):
    """
    Real-time WebSocket endpoint for PC / Server Mode inference.
    Routes frames through the modular perception and guidance pipeline:
    YOLO26 -> Depth -> Spatial Fusion -> Corridor -> Tracking -> Risk -> Priority -> Guidance.
    """
    await websocket.accept()
    print("[WebSocket] Phone connected for PC Inference Mode.")

    active_model = "base_yolo26n"
    resolution = 480
    depth_model = "yolo26n_depth"
    depth_resolution = 512
    enable_depth = True
    confidence = 0.25
    record_enabled = False
    record_freq = 5
    only_detections = False

    try:
        while True:
            message = await websocket.receive()

            if "text" in message:
                try:
                    data = json.loads(message["text"])
                except Exception:
                    continue

                msg_type = data.get("type")

                # 1. Configuration message
                if msg_type == "configure":
                    active_model = data.get("model", active_model)
                    resolution = int(data.get("resolution", resolution))
                    depth_model = data.get("depth_model", depth_model)
                    depth_resolution = int(data.get("depth_resolution", depth_resolution))
                    enable_depth = bool(data.get("enable_depth", True))
                    confidence = float(data.get("confidence", confidence))

                    rec_config = data.get("record", {})
                    record_enabled = bool(rec_config.get("enabled", False))
                    record_freq = int(rec_config.get("frequency", 5))
                    only_detections = bool(rec_config.get("only_detections", False))

                    # Configure master navigation pipeline
                    loaded = navigation_pipeline.configure(
                        yolo_model_id=active_model,
                        yolo_resolution=resolution,
                        depth_model_id=depth_model,
                        depth_resolution=depth_resolution,
                        enable_depth=enable_depth
                    )

                    session_id = None
                    if record_enabled:
                        session_id = session_recorder.start_session(
                            model_name=f"{active_model} + {depth_model}({depth_resolution}p)",
                            resolution=resolution,
                            confidence=confidence,
                            enabled=record_enabled,
                            frequency=record_freq,
                            only_detections=only_detections,
                            device=f"PC ({navigation_pipeline.detector.device.upper()})"
                        )

                    await websocket.send_text(json.dumps({
                        "type": "configured",
                        "status": "ready" if loaded else "error",
                        "model": active_model,
                        "resolution": resolution,
                        "depth_model": depth_model,
                        "depth_resolution": depth_resolution,
                        "device": navigation_pipeline.detector.device.upper(),
                        "classes": len(navigation_pipeline.detector.class_names),
                        "session_id": session_id
                    }))
                    continue

                # 2. Stop/End session message
                elif msg_type == "stop_session":
                    summary = session_recorder.end_session()
                    await websocket.send_text(json.dumps({
                        "type": "session_ended",
                        "summary": summary
                    }))
                    continue

                # 3. Frame message (base64 image)
                elif msg_type == "frame":
                    frame_id = int(data.get("frame_id", 0))
                    timestamp = int(data.get("timestamp", int(time.time() * 1000)))
                    image_b64 = data.get("image", "")

                    if not image_b64:
                        continue

                    if "," in image_b64:
                        image_b64 = image_b64.split(",", 1)[1]

                    try:
                        image_bytes = base64.b64decode(image_b64)
                    except Exception:
                        continue

                    np_arr = np.frombuffer(image_bytes, np.uint8)
                    frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if frame_bgr is None:
                        continue

                    try:
                        result_payload, annotated_bgr = navigation_pipeline.process_frame(
                            frame_bgr=frame_bgr,
                            confidence_threshold=confidence,
                            frame_id=frame_id
                        )
                    except Exception as e:
                        print(f"[WebSocket] Pipeline error on frame {frame_id}: {e}")
                        continue

                    # Record session frame if enabled
                    if record_enabled:
                        session_recorder.record_frame(
                            frame_id=frame_id,
                            timestamp_client=timestamp,
                            original_bgr=frame_bgr,
                            detections=result_payload["objects"],
                            fps=result_payload["fps"],
                            latency_dict=result_payload["latency"],
                            annotated_bgr=annotated_bgr,
                            result_payload=result_payload
                        )

                    response_payload = {
                        "type": "result",
                        "frame_id": frame_id,
                        "timestamp": timestamp,
                        "fps": result_payload["fps"],
                        "latency": result_payload["latency"],
                        "objects": result_payload["objects"],
                        "events": result_payload["events"],
                        "detections": result_payload["objects"], # backward compatibility
                        "count": len(result_payload["objects"])
                    }
                    await websocket.send_text(json.dumps(response_payload))

            elif "bytes" in message:
                # Binary frame protocol: First 4 bytes uint32 frame_id, next 8 bytes uint64 timestamp, rest is JPEG
                raw_bytes = message["bytes"]
                if len(raw_bytes) < 12:
                    continue
                frame_id, timestamp = struct.unpack(">IQ", raw_bytes[:12])
                jpeg_bytes = raw_bytes[12:]

                np_arr = np.frombuffer(jpeg_bytes, np.uint8)
                frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame_bgr is None:
                    continue

                try:
                    result_payload, annotated_bgr = navigation_pipeline.process_frame(
                        frame_bgr=frame_bgr,
                        confidence_threshold=confidence,
                        frame_id=frame_id
                    )
                except Exception as e:
                    print(f"[WebSocket] Binary pipeline error: {e}")
                    continue

                if record_enabled:
                    session_recorder.record_frame(
                        frame_id=frame_id,
                        timestamp_client=timestamp,
                        original_bgr=frame_bgr,
                        detections=result_payload["objects"],
                        fps=result_payload["fps"],
                        latency_dict=result_payload["latency"],
                        annotated_bgr=annotated_bgr,
                        result_payload=result_payload
                    )

                response_payload = {
                    "type": "result",
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "fps": result_payload["fps"],
                    "latency": result_payload["latency"],
                    "objects": result_payload["objects"],
                    "events": result_payload["events"],
                    "detections": result_payload["objects"],
                    "count": len(result_payload["objects"])
                }
                await websocket.send_text(json.dumps(response_payload))

    except WebSocketDisconnect:
        print("[WebSocket] Phone client disconnected.")
        session_recorder.end_session()
    except Exception as e:
        print(f"[WebSocket] Error: {e}")
        session_recorder.end_session()

