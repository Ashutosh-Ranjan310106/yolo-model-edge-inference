package com.drishtix.rod.ui

import android.graphics.Paint
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import com.drishtix.rod.inference.Detection

@Composable
fun BoundingBoxOverlay(
    detections: List<Detection>,
    frameWidth: Int,
    frameHeight: Int,
    modifier: Modifier = Modifier
) {
    Canvas(modifier = modifier.fillMaxSize()) {
        if (frameWidth <= 0 || frameHeight <= 0) return@Canvas

        val scaleX = size.width / frameWidth.toFloat()
        val scaleY = size.height / frameHeight.toFloat()

        val textPaint = Paint().apply {
            color = android.graphics.Color.BLACK
            textSize = 36f
            isFakeBoldText = true
        }

        for (det in detections) {
            val left = det.box.left * scaleX
            val top = det.box.top * scaleY
            val right = det.box.right * scaleX
            val bottom = det.box.bottom * scaleY
            val boxW = right - left
            val boxH = bottom - top

            // Hazard category color coding
            val lower = det.className.lowercase()
            val boxColor = when {
                lower.contains("stairs") || lower.contains("manhole") -> Color(0xFFEF4444) // Red
                lower.contains("car") || lower.contains("person") || lower.contains("bus") || lower.contains("bike") -> Color(0xFFF59E0B) // Amber
                lower.contains("door") || lower.contains("wall") -> Color(0xFFA855F7) // Purple
                else -> Color(0xFF38BDF8) // Cyan
            }

            // Draw Box
            drawRect(
                color = boxColor,
                topLeft = Offset(left, top),
                size = Size(boxW, boxH),
                style = Stroke(width = 6f)
            )

            // Draw Header Badge with Class, Score, Distance
            val distText = det.distance?.displayText ?: ""
            val label = "${det.className.uppercase()} ${(det.score * 100).toInt()}% ${if (distText.isNotEmpty()) "| $distText" else ""}"
            val textWidth = textPaint.measureText(label)
            val badgeHeight = 48f

            drawRect(
                color = boxColor,
                topLeft = Offset(left, (top - badgeHeight).coerceAtLeast(0f)),
                size = Size(textWidth + 24f, badgeHeight)
            )

            drawContext.canvas.nativeCanvas.drawText(
                label,
                left + 12f,
                (top - badgeHeight + 36f).coerceAtLeast(36f),
                textPaint
            )
        }
    }
}
