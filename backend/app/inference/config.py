"""
Centralized Configuration for Navigation Perception & Guidance Pipeline.
Contains all thresholds, corridor geometry, criticality values, risk factors,
priority tiers, and guidance templates without magic numbers.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass, field

@dataclass
class DirectionConfig:
    # 7-zone directional division based on normalized relative_x in [-1.0, 1.0]
    # relative_x = (center_x - width / 2) / (width / 2)
    far_left_max: float = -0.70
    left_max: float = -0.35
    slightly_left_max: float = -0.12
    center_max: float = 0.12
    slightly_right_max: float = 0.35
    right_max: float = 0.70

@dataclass
class DistanceConfig:
    # Metric distance thresholds in meters
    very_near_max_m: float = 1.0
    near_max_m: float = 2.0
    medium_max_m: float = 5.0
    far_max_m: float = 10.0

    # Sampling parameters inside YOLO bounding box
    roi_crop_x_margin: float = 0.15   # ignore outer 15% on left/right (use central 70%)
    roi_crop_y_margin: float = 0.15   # ignore outer 15% on top/bottom (use central 70%)
    min_valid_samples: int = 5

    # Fallback bounding-box area ratio thresholds (if depth unavailable)
    bbox_ratio_very_near: float = 0.35
    bbox_ratio_near: float = 0.20
    bbox_ratio_medium: float = 0.10
    bbox_ratio_far: float = 0.05

@dataclass
class CorridorConfig:
    # Trapezoidal walking corridor from bottom-center of image toward horizon
    # Coordinates normalized [0.0, 1.0]
    horizon_y: float = 0.45       # Top of the corridor (closer to horizon)
    bottom_y: float = 1.00        # Bottom of image (user's feet)
    top_width_ratio: float = 0.30 # Width of trapezoid at horizon_y (30% of image width)
    bottom_width_ratio: float = 0.80 # Width of trapezoid at bottom_y (80% of image width)

    # Overlap ratio classifications
    outside_threshold: float = 0.20    # 0.0 - 0.20: outside path
    partial_threshold: float = 0.50    # 0.20 - 0.50: partially relevant
    # 0.50 - 1.00: likely in path

@dataclass
class TrackingConfig:
    iou_threshold: float = 0.25        # IoU matching threshold
    max_frames_lost: int = 5          # Number of frames to maintain lost tracks
    track_timeout_sec: float = 0.8    # Time to keep track alive without detection
    # Exponential Moving Average (EMA) smoothing factors (0.0 < alpha <= 1.0)
    # smoothed = (1 - alpha) * prev + alpha * curr
    alpha_x: float = 0.35
    alpha_distance: float = 0.35
    alpha_box: float = 0.50

@dataclass
class RiskConfig:
    # Base criticality table per object class
    criticality_table: Dict[str, float] = field(default_factory=lambda: {
        "manhole": 1.00,
        "stairs": 1.00,
        "car": 0.95,
        "bus": 0.95,
        "truck": 0.95,
        "motorcycle": 0.90,
        "bike": 0.85,
        "person": 0.70,
        "dog": 0.70,
        "guard rail": 0.70,
        "guardrail": 0.70,
        "electrical pole": 0.70,
        "pole": 0.70,
        "traffic sign": 0.60,
        "traffic cone": 0.65,
        "cone": 0.65,
        "door": 0.55,
        "wall": 0.50,
        "plant pot": 0.45,
        "potted plant": 0.45,
        "chair": 0.35,
        "bench": 0.30,
        "unknown": 0.75,
        "unidentified obstacle": 0.75
    })
    default_criticality: float = 0.50

    # Distance factor multipliers
    distance_factors: Dict[str, float] = field(default_factory=lambda: {
        "VERY_NEAR": 1.00,
        "NEAR": 0.80,
        "MEDIUM": 0.50,
        "FAR": 0.20,
        "VERY_FAR": 0.05
    })

    # Path/corridor relevance multipliers
    path_factors: Dict[str, float] = field(default_factory=lambda: {
        "directly in path": 1.00,
        "partially in path": 0.70,
        "near path": 0.40,
        "outside path": 0.10
    })

    # Track persistence multipliers based on consecutive frames seen
    persistence_factors: Dict[int, float] = field(default_factory=lambda: {
        1: 0.40,
        2: 0.70,
        3: 1.00
    })

    # Priority score cutoffs
    priority_critical_threshold: float = 0.75
    priority_high_threshold: float = 0.50
    priority_medium_threshold: float = 0.25
    priority_low_threshold: float = 0.10

@dataclass
class PerceptionConfig:
    # Ground drop/rise analysis parameters
    ground_band_y_start: float = 0.65   # Analyze lower 35% of depth map
    ground_band_y_end: float = 0.95
    drop_depth_jump_ratio: float = 1.6  # Sudden increase in depth indicates drop/hole
    rise_depth_jump_ratio: float = 0.6  # Sudden decrease in depth indicates rise/step
    stair_gradient_variance: float = 0.25

    # Unknown obstacle parameters
    unknown_min_blob_area_ratio: float = 0.02 # Must be at least 2% of corridor area
    unknown_min_depth_m: float = 0.5
    unknown_max_depth_m: float = 4.0
    unknown_corridor_relevance_min: float = 0.30

    # Wall analysis parameters
    wall_min_corridor_overlap: float = 0.35
    wall_max_announce_distance_m: float = 2.5

@dataclass
class GuidanceConfig:
    # Event cooldowns to prevent repetitive speech
    same_event_cooldown_sec: float = 3.0
    direction_change_cooldown_sec: float = 1.5
    distance_change_cooldown_sec: float = 1.5
    critical_hazard_cooldown_sec: float = 1.0
    max_concurrent_announcements: int = 1

@dataclass
class PipelineConfig:
    direction: DirectionConfig = field(default_factory=DirectionConfig)
    distance: DistanceConfig = field(default_factory=DistanceConfig)
    corridor: CorridorConfig = field(default_factory=CorridorConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)

# Global default instance
pipeline_config = PipelineConfig()
