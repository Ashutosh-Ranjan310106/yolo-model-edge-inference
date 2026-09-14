package com.drishtix.rod.ui

import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import com.drishtix.rod.inference.InferenceResult

@Composable
fun LiveDetectionScreen(
    previewView: PreviewView,
    result: InferenceResult?,
    frameWidth: Int,
    frameHeight: Int,
    modelTag: String,
    onStopClick: () -> Unit
) {
    Box(modifier = Modifier.fillMaxSize().background(Color.Black)) {
        // 1. Camera Preview
        AndroidView(
            factory = { previewView },
            modifier = Modifier.fillMaxSize()
        )

        // 2. Bounding Box & Distance Canvas Overlay
        result?.let {
            BoundingBoxOverlay(
                detections = it.detections,
                frameWidth = frameWidth,
                frameHeight = frameHeight,
                modifier = Modifier.fillMaxSize()
            )
        }

        // 3. Top Floating Performance HUD
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            // Main HUD Pill
            Surface(
                color = Color(0xCC0F172A),
                shape = RoundedCornerShape(8.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                Row(
                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        "$modelTag [${result?.provider ?: "CPU"}]",
                        color = Color(0xFF38BDF8),
                        fontWeight = FontWeight.Bold,
                        fontFamily = FontFamily.Monospace,
                        fontSize = 13.sp
                    )
                    Text(
                        "FPS: %.1f".format(result?.fps ?: 0.0f),
                        color = Color(0xFF10B981),
                        fontWeight = FontWeight.Bold,
                        fontFamily = FontFamily.Monospace,
                        fontSize = 13.sp
                    )
                    Text(
                        "Objs: ${result?.detections?.size ?: 0}",
                        color = Color.White,
                        fontWeight = FontWeight.Bold,
                        fontFamily = FontFamily.Monospace,
                        fontSize = 13.sp
                    )
                }
            }

            // Latency Breakdown Pill
            result?.let {
                Surface(
                    color = Color(0xAA0F172A),
                    shape = RoundedCornerShape(6.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Row(
                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text("Pre: ${it.latency.preprocessMs}ms", color = Color.Gray, fontSize = 11.sp, fontFamily = FontFamily.Monospace)
                        Text("Inf: ${it.latency.inferenceMs}ms", color = Color.Gray, fontSize = 11.sp, fontFamily = FontFamily.Monospace)
                        Text("Post: ${it.latency.postprocessMs}ms", color = Color.Gray, fontSize = 11.sp, fontFamily = FontFamily.Monospace)
                        Text("Total: ${it.latency.totalMs}ms", color = Color(0xFF38BDF8), fontSize = 11.sp, fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold)
                    }
                }
            }
        }

        // 4. Bottom Stop Button
        Button(
            onClick = onStopClick,
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(bottom = 32.dp),
            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFEF4444)),
            shape = CircleShape
        ) {
            Text("⏹ STOP DETECTION", fontWeight = FontWeight.Bold)
        }
    }
}
