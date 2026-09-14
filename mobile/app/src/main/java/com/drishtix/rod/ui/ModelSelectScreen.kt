package com.drishtix.rod.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.drishtix.rod.model.RemoteModel

@Composable
fun ModelSelectScreen(
    serverUrl: String,
    models: List<RemoteModel>,
    selectedModel: RemoteModel?,
    selectedResolution: Int,
    isDownloaded: Boolean,
    isDownloading: Boolean,
    downloadProgress: Int,
    confThreshold: Float,
    onServerUrlChange: (String) -> Unit,
    onRefreshModels: () -> Unit,
    onModelSelect: (RemoteModel) -> Unit,
    onResolutionChange: (Int) -> Unit,
    onConfidenceChange: (Float) -> Unit,
    onDownloadClick: () -> Unit,
    onStartClick: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF090D16))
            .padding(16.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // App Header
        Row(
            modifier = Modifier.fillMaxWidth().padding(top = 16.dp, bottom = 8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text("🎯", fontSize = 28.sp)
            Spacer(modifier = Modifier.width(8.dp))
            Text(
                "ROD EDGE DETECTOR",
                color = Color.White,
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold
            )
        }

        // Server Connection Card
        Card(
            colors = CardDefaults.cardColors(containerColor = Color(0xFF131B2E)),
            shape = RoundedCornerShape(12.dp)
        ) {
            Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("BACKEND SERVER (LAN)", color = Color(0xFF38BDF8), fontWeight = FontWeight.SemiBold)
                OutlinedTextField(
                    value = serverUrl,
                    onValueChange = onServerUrlChange,
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text("Server Base URL") },
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedTextColor = Color.White,
                        unfocusedTextColor = Color.White,
                        focusedBorderColor = Color(0xFF38BDF8)
                    )
                )
                Button(
                    onClick = onRefreshModels,
                    modifier = Modifier.fillMaxWidth(),
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF1E293B))
                ) {
                    Text("Connect & Fetch Models (${models.size} found)")
                }
            }
        }

        // Model Selection Card
        Card(
            colors = CardDefaults.cardColors(containerColor = Color(0xFF131B2E)),
            shape = RoundedCornerShape(12.dp)
        ) {
            Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text("MODEL & CHECKPOINT SELECTION", color = Color(0xFF38BDF8), fontWeight = FontWeight.SemiBold)

                // Input Resolution Selector
                Text("Input Resolution:", color = Color.White, fontSize = 14.sp)
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    listOf(320, 480, 640).forEach { res ->
                        val isResSelected = selectedResolution == res
                        Button(
                            onClick = { onResolutionChange(res) },
                            modifier = Modifier.weight(1f),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = if (isResSelected) Color(0xFF0284C7) else Color(0xFF1E293B)
                            )
                        ) {
                            Text("${res}x${res}")
                        }
                    }
                }

                Text("Available Training Checkpoints (${models.size}):", color = Color.Gray, fontSize = 13.sp)
                models.forEach { model ->
                    val isSelected = selectedModel?.id == model.id
                    Button(
                        onClick = { onModelSelect(model) },
                        modifier = Modifier.fillMaxWidth(),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = if (isSelected) Color(0xFF0284C7) else Color(0xFF1E293B)
                        )
                    ) {
                        Text(
                            "${model.name} (~${model.fileSizeMb}MB)",
                            fontSize = 13.sp
                        )
                    }
                }

                // Confidence Slider
                Text("Confidence Cutoff: ${(confThreshold * 100).toInt()}%", color = Color.White)
                Slider(
                    value = confThreshold,
                    onValueChange = onConfidenceChange,
                    valueRange = 0.1f..0.9f,
                    steps = 15,
                    colors = SliderDefaults.colors(thumbColor = Color(0xFF38BDF8), activeTrackColor = Color(0xFF0284C7))
                )
            }
        }

        // Cache & Download Status Card
        if (selectedModel != null) {
            Card(
                colors = CardDefaults.cardColors(containerColor = Color(0xFF131B2E)),
                shape = RoundedCornerShape(12.dp)
            ) {
                Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("LOCAL CACHE STATUS", color = Color(0xFF38BDF8), fontWeight = FontWeight.SemiBold)
                    Text("Selected: ${selectedModel.name}", color = Color.White, fontWeight = FontWeight.Medium)
                    Text("Resolution: ${selectedResolution}x${selectedResolution}", color = Color.Gray)
                    Text(
                        if (isDownloaded) "✓ Model Cached on Phone" else "✗ Not Downloaded Yet",
                        color = if (isDownloaded) Color(0xFF10B981) else Color(0xFFF59E0B),
                        fontWeight = FontWeight.Bold
                    )

                    if (isDownloading) {
                        LinearProgressIndicator(
                            progress = downloadProgress / 100f,
                            modifier = Modifier.fillMaxWidth().height(8.dp),
                            color = Color(0xFF38BDF8)
                        )
                        Text("Exporting & Downloading... $downloadProgress%", color = Color.Gray, fontSize = 12.sp)
                    }

                    Button(
                        onClick = onDownloadClick,
                        modifier = Modifier.fillMaxWidth(),
                        enabled = !isDownloading,
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF1E293B))
                    ) {
                        Text(if (isDownloaded) "Re-download Model" else "⬇ Download Model (${selectedResolution}x${selectedResolution})")
                    }
                }
            }
        }

        // Start Live Detection Action
        Button(
            onClick = onStartClick,
            modifier = Modifier.fillMaxWidth().height(56.dp),
            enabled = selectedModel != null && isDownloaded && !isDownloading,
            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF0284C7)),
            shape = RoundedCornerShape(14.dp)
        ) {
            Text("🚀 START LIVE DETECTION", fontSize = 16.sp, fontWeight = FontWeight.Bold)
        }
    }
}
