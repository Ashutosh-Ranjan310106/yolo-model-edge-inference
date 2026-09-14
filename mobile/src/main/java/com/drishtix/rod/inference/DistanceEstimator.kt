package com.drishtix.rod.inference

import android.graphics.RectF

class DistanceEstimator {

    data class Tier(
        val minRatio: Float,
        val maxRatio: Float,
        val label: String,
        val approxMeters: Float
    )

    private val tiers = listOf(
        Tier(0.35f, 1.00f, "<1 m", 0.8f),
        Tier(0.20f, 0.35f, "1–2 m", 1.5f),
        Tier(0.10f, 0.20f, "3–5 m", 3.5f),
        Tier(0.05f, 0.10f, "6–10 m", 7.5f),
        Tier(0.00f, 0.05f, ">10 m", 12.0f)
    )

    fun estimate(box: RectF, frameWidth: Int, frameHeight: Int): DistanceEstimate {
        val bboxArea = box.width() * box.height()
        val frameArea = (frameWidth * frameHeight).coerceAtLeast(1).toFloat()
        val ratio = (bboxArea / frameArea).coerceIn(0.0f, 1.0f)

        for (tier in tiers) {
            if (ratio >= tier.minRatio && ratio <= tier.maxRatio) {
                return DistanceEstimate(
                    label = tier.label,
                    approxMeters = tier.approxMeters,
                    areaRatio = ratio,
                    displayText = "~${tier.approxMeters}m"
                )
            }
        }

        return DistanceEstimate(
            label = ">10 m",
            approxMeters = 12.0f,
            areaRatio = ratio,
            displayText = ">10m"
        )
    }

    fun estimateFromDepth(depthMeters: Float, areaRatio: Float = 0.0f): DistanceEstimate {
        val label = when {
            depthMeters < 1.0f -> "<1 m"
            depthMeters < 2.0f -> "1–2 m"
            depthMeters < 5.0f -> "3–5 m"
            depthMeters < 10.0f -> "6–10 m"
            else -> ">10 m"
        }
        return DistanceEstimate(
            label = label,
            approxMeters = depthMeters,
            areaRatio = areaRatio,
            displayText = "${String.format("%.1f", depthMeters)}m"
        )
    }
}
