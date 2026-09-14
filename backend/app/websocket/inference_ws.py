import base64
import json
import time
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.inference.pc_runner import pc_runner
from app.services.session_recorder import session_recorder

router = APIRouter(tags=["websocket"])

@router.websocket("/ws/inference")
async def websocket_inference_endpoint(websocket: WebSocket):
    """
    Real-time WebSocket endpoint for PC / Server Mode inference.
    Receives camera frames from phone, executes PyTorch YOLO .pt inference,
    records sessions if enabled, and streams back detection JSON.
    """
    await websocket.accept()
    print("[WebSocket] Phone connected for PC Inference Mode.")

    active_model = "base_yolo26n"
    resolution = 480
    confidence = 0.25
    record_enabled = False
    record_freq = 5
    only_detections = False

    # FPS tracking on PC side
    frame_times = []
    fps = 0.0

    try:
        while True:
            # Receive message (supports both text JSON and binary)
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
                    confidence = float(data.get("confidence", confidence))

                    rec_config = data.get("record", {})
                    record_enabled = bool(rec_config.get("enabled", False))
                    record_freq = int(rec_config.get("frequency", 5))
                    only_detections = bool(rec_config.get("only_detections", False))

                    # Load model into PC GPU/CPU memory
                    loaded = pc_runner.load_model(active_model)
                    
                    session_id = None
                    if record_enabled:
                        session_id = session_recorder.start_session(
                            model_name=active_model,
                            resolution=resolution,
                            confidence=confidence,
                            enabled=record_enabled,
                            frequency=record_freq,
                            only_detections=only_detections,
                            device=f"PC ({pc_runner.device.upper()})"
                        )

                    await websocket.send_text(json.dumps({
                        "type": "configured",
                        "status": "ready" if loaded else "error",
                        "model": active_model,
                        "resolution": resolution,
                        "device": pc_runner.device.upper(),
                        "classes": len(pc_runner.class_names),
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

                    # Strip data URL prefix if present
                    if "," in image_b64:
                        image_b64 = image_b64.split(",", 1)[1]

                    try:
                        image_bytes = base64.b64decode(image_b64)
                    except Exception:
                        continue

                    t_start_infer = time.perf_counter()

                    try:
                        frame_bgr, detections, latencies = pc_runner.predict_frame(
                            image_bytes=image_bytes,
                            resolution=resolution,
                            confidence_threshold=confidence
                        )
                    except Exception as e:
                        print(f"[WebSocket] Inference error on frame {frame_id}: {e}")
                        continue

                    # Calculate rolling FPS
                    now = time.perf_counter()
                    frame_times.append(now)
                    if len(frame_times) > 15:
                        frame_times.pop(0)
                    if len(frame_times) >= 2:
                        duration = frame_times[-1] - frame_times[0]
                        fps = round((len(frame_times) - 1) / duration, 1) if duration > 0 else 0.0

                    # Record session frame if enabled
                    if record_enabled:
                        session_recorder.record_frame(
                            frame_id=frame_id,
                            timestamp_client=timestamp,
                            original_bgr=frame_bgr,
                            detections=detections,
                            fps=fps,
                            latency_dict=latencies
                        )

                    # Return result
                    response_payload = {
                        "type": "result",
                        "frame_id": frame_id,
                        "timestamp": timestamp,
                        "fps": fps,
                        "latency": latencies,
                        "detections": detections,
                        "count": len(detections)
                    }
                    await websocket.send_text(json.dumps(response_payload))

            elif "bytes" in message:
                # Binary frame protocol: First 4 bytes uint32 frame_id, next 8 bytes uint64 timestamp, rest is JPEG
                raw_bytes = message["bytes"]
                if len(raw_bytes) < 12:
                    continue
                import struct
                frame_id, timestamp = struct.unpack(">IQ", raw_bytes[:12])
                jpeg_bytes = raw_bytes[12:]

                try:
                    frame_bgr, detections, latencies = pc_runner.predict_frame(
                        image_bytes=jpeg_bytes,
                        resolution=resolution,
                        confidence_threshold=confidence
                    )
                except Exception as e:
                    print(f"[WebSocket] Binary inference error: {e}")
                    continue

                now = time.perf_counter()
                frame_times.append(now)
                if len(frame_times) > 15:
                    frame_times.pop(0)
                if len(frame_times) >= 2:
                    duration = frame_times[-1] - frame_times[0]
                    fps = round((len(frame_times) - 1) / duration, 1) if duration > 0 else 0.0

                if record_enabled:
                    session_recorder.record_frame(
                        frame_id=frame_id,
                        timestamp_client=timestamp,
                        original_bgr=frame_bgr,
                        detections=detections,
                        fps=fps,
                        latency_dict=latencies
                    )

                response_payload = {
                    "type": "result",
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "fps": fps,
                    "latency": latencies,
                    "detections": detections,
                    "count": len(detections)
                }
                await websocket.send_text(json.dumps(response_payload))

    except WebSocketDisconnect:
        print("[WebSocket] Phone client disconnected.")
        session_recorder.end_session()
    except Exception as e:
        print(f"[WebSocket] Error: {e}")
        session_recorder.end_session()
