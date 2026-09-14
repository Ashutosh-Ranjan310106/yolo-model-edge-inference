package com.drishtix.rod.inference

import android.graphics.RectF

data class DistanceEstimate(
    val label: String,
    val approxMeters: Float,
    val areaRatio: Float,
    val displayText: String
)

data class Detection(
    val box: RectF,
    val score: Float,
    val classId: Int,
    val className: String,
    val distance: DistanceEstimate? = null
)

data class LatencyMetrics(
    val preprocessMs: Long = 0,
    val inferenceMs: Long = 0,
    val postprocessMs: Long = 0,
    val totalMs: Long = 0
)

data class InferenceResult(
    val detections: List<Detection>,
    val latency: LatencyMetrics,
    val fps: Float,
    val provider: String = "CPU"
)
