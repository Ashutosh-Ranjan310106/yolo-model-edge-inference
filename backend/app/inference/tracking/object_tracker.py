"""
Object Tracker Module.
Lightweight temporal tracking using IoU association and persistence tracking.
Smooths position and distance using TemporalSmoother.
"""

from typing import List, Dict, Any, Optional
import time
from dataclasses import dataclass, field

from app.inference.config import TrackingConfig, pipeline_config
from app.inference.spatial.spatial_fusion import SpatialObject
from app.inference.tracking.temporal_smoother import TemporalSmoother

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculates Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    xa = max(box1[0], box2[0])
    ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2])
    yb = min(box1[3], box2[3])

    inter_w = max(0.0, xb - xa)
    inter_h = max(0.0, yb - ya)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union

@dataclass
class Track:
    track_id: int
    class_name: str
    last_bbox: List[float]
    direction: str
    horizontal_position: float
    distance_m: Optional[float]
    distance_category: str
    corridor_overlap: float
    path_relevance: str
    confidence: float
    depth_confidence: float
    type: str

    frames_seen: int = 1
    frames_lost: int = 0
    last_seen_time: float = field(default_factory=time.time)

    direction_history: List[str] = field(default_factory=list)
    distance_history: List[float] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)

class ObjectTracker:
    """Tracks spatial objects across consecutive video frames using IoU association."""
    def __init__(self, config: TrackingConfig = None):
        self.config = config or pipeline_config.tracking
        self.smoother = TemporalSmoother(self.config)
        self.active_tracks: List[Track] = []
        self.next_track_id: int = 1

    def update(self, incoming_objects: List[SpatialObject]) -> List[SpatialObject]:
        """
        Associates incoming detections with active tracks, smooths metrics, and updates persistence.
        Returns stabilized, tracked SpatialObject list.
        """
        now = time.time()
        cfg = self.config

        unmatched_incoming = set(range(len(incoming_objects)))
        matched_tracks = set()

        # 1. IoU Association: Match active tracks with current detections of same class
        for track in self.active_tracks:
            best_iou = 0.0
            best_idx = None

            for idx in unmatched_incoming:
                inc = incoming_objects[idx]
                if inc.class_name == track.class_name or inc.type == track.type:
                    iou = calculate_iou(track.last_bbox, inc.bbox)
                    if iou > cfg.iou_threshold and iou > best_iou:
                        best_iou = iou
                        best_idx = idx

            if best_idx is not None:
                inc = incoming_objects[best_idx]
                unmatched_incoming.remove(best_idx)
                matched_tracks.add(track.track_id)

                # Smooth bounding box
                smoothed_box = []
                for k in range(4):
                    b_val = (1.0 - cfg.alpha_box) * track.last_bbox[k] + cfg.alpha_box * inc.bbox[k]
                    smoothed_box.append(round(b_val, 1))
                track.last_bbox = smoothed_box

                # Smooth continuous horizontal position and derive direction
                s_h_pos, s_dir = self.smoother.smooth_position(
                    track.horizontal_position, inc.horizontal_position
                )
                track.horizontal_position = s_h_pos
                track.direction = s_dir

                # Smooth distance in meters and derive category
                s_dist, s_cat = self.smoother.smooth_distance(
                    track.distance_m, inc.distance_m
                )
                track.distance_m = s_dist
                track.distance_category = s_cat

                # Update confidence & tracking history
                track.confidence = round(0.5 * track.confidence + 0.5 * inc.confidence, 2)
                track.corridor_overlap = inc.corridor_overlap
                track.path_relevance = inc.path_relevance
                track.depth_confidence = inc.depth_confidence
                track.frames_seen += 1
                track.frames_lost = 0
                track.last_seen_time = now

                track.direction_history.append(s_dir)
                if len(track.direction_history) > 10:
                    track.direction_history.pop(0)

                if s_dist is not None:
                    track.distance_history.append(s_dist)
                    if len(track.distance_history) > 10:
                        track.distance_history.pop(0)

                track.confidence_history.append(inc.confidence)
                if len(track.confidence_history) > 10:
                    track.confidence_history.pop(0)

        # 2. Add newly appeared objects as new tracks
        for idx in unmatched_incoming:
            inc = incoming_objects[idx]
            new_track = Track(
                track_id=self.next_track_id,
                class_name=inc.class_name,
                last_bbox=inc.bbox,
                direction=inc.direction,
                horizontal_position=inc.horizontal_position,
                distance_m=inc.distance_m,
                distance_category=inc.distance_category,
                corridor_overlap=inc.corridor_overlap,
                path_relevance=inc.path_relevance,
                confidence=inc.confidence,
                depth_confidence=inc.depth_confidence,
                type=inc.type,
                frames_seen=1,
                frames_lost=0,
                last_seen_time=now,
                direction_history=[inc.direction],
                distance_history=[inc.distance_m] if inc.distance_m is not None else [],
                confidence_history=[inc.confidence]
            )
            self.next_track_id += 1
            self.active_tracks.append(new_track)

        # 3. Update lost tracks and prune expired ones
        pruned_tracks: List[Track] = []
        for track in self.active_tracks:
            if track.track_id not in matched_tracks and track.track_id not in [self.next_track_id - 1]:
                track.frames_lost += 1

            time_elapsed = now - track.last_seen_time
            if track.frames_lost <= cfg.max_frames_lost and time_elapsed <= cfg.track_timeout_sec:
                pruned_tracks.append(track)

        self.active_tracks = pruned_tracks

        # 4. Construct output SpatialObjects
        output_objects: List[SpatialObject] = []
        for track in self.active_tracks:
            # Only output tracks that are currently visible or briefly coasted
            obj = SpatialObject(
                id=track.track_id,
                type=track.type,
                class_name=track.class_name,
                confidence=track.confidence,
                bbox=track.last_bbox,
                direction=track.direction,
                horizontal_position=track.horizontal_position,
                distance_m=track.distance_m,
                distance_category=track.distance_category,
                corridor_overlap=track.corridor_overlap,
                path_relevance=track.path_relevance,
                depth_confidence=track.depth_confidence,
                track_id=track.track_id,
                frames_seen=track.frames_seen
            )
            output_objects.append(obj)

        return output_objects
