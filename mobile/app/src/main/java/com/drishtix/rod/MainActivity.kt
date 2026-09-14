package com.drishtix.rod

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.runtime.*
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.drishtix.rod.camera.CameraAnalyzer
import com.drishtix.rod.inference.InferenceResult
import com.drishtix.rod.inference.YoloDetector
import com.drishtix.rod.model.LocalModelCache
import com.drishtix.rod.model.ModelRegistryClient
import com.drishtix.rod.model.RemoteModel
import com.drishtix.rod.ui.LiveDetectionScreen
import com.drishtix.rod.ui.ModelSelectScreen
import kotlinx.coroutines.launch
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {

    private lateinit var modelCache: LocalModelCache
    private lateinit var registryClient: ModelRegistryClient
    private val cameraExecutor = Executors.newSingleThreadExecutor()

    private var activeDetector: YoloDetector? = null

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (!isGranted) {
            Toast.makeText(this, "Camera permission is required for real-time edge inference", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        modelCache = LocalModelCache(this)
        registryClient = ModelRegistryClient("http://192.168.1.100:8000")

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }

        setContent {
            var isLive by remember { mutableStateOf(false) }
            var serverUrl by remember { mutableStateOf("http://192.168.1.100:8000") }
            var models by remember { mutableStateOf<List<RemoteModel>>(emptyList()) }
            var selectedModel by remember { mutableStateOf<RemoteModel?>(null) }
            var selectedResolution by remember { mutableIntStateOf(480) }
            var isDownloaded by remember { mutableStateOf(false) }
            var isDownloading by remember { mutableStateOf(false) }
            var downloadProgress by remember { mutableIntStateOf(0) }
            var confThreshold by remember { mutableFloatStateOf(0.25f) }

            // Live state
            var currentResult by remember { mutableStateOf<InferenceResult?>(null) }
            var frameW by remember { mutableIntStateOf(0) }
            var frameH by remember { mutableIntStateOf(0) }

            val previewView = remember { PreviewView(this) }

            fun checkDownloaded(model: RemoteModel?, res: Int): Boolean {
                if (model == null) return false
                val key = "${model.id}_$res"
                return modelCache.isModelDownloaded(key) || modelCache.isModelDownloaded(model.id)
            }

            LaunchedEffect(Unit) {
                fetchModels(serverUrl) { list ->
                    models = list
                    if (list.isNotEmpty()) {
                        selectedModel = list.first()
                        isDownloaded = checkDownloaded(list.first(), selectedResolution)
                    }
                }
            }

            if (!isLive) {
                ModelSelectScreen(
                    serverUrl = serverUrl,
                    models = models,
                    selectedModel = selectedModel,
                    selectedResolution = selectedResolution,
                    isDownloaded = isDownloaded,
                    isDownloading = isDownloading,
                    downloadProgress = downloadProgress,
                    confThreshold = confThreshold,
                    onServerUrlChange = { serverUrl = it; registryClient.updateServerUrl(it) },
                    onRefreshModels = {
                        fetchModels(serverUrl) { list ->
                            models = list
                            if (list.isNotEmpty()) {
                                selectedModel = list.first()
                                isDownloaded = checkDownloaded(list.first(), selectedResolution)
                            }
                        }
                    },
                    onModelSelect = { model ->
                        selectedModel = model
                        isDownloaded = checkDownloaded(model, selectedResolution)
                    },
                    onResolutionChange = { res ->
                        selectedResolution = res
                        isDownloaded = checkDownloaded(selectedModel, res)
                    },
                    onConfidenceChange = { confThreshold = it },
                    onDownloadClick = {
                        selectedModel?.let { model ->
                            isDownloading = true
                            lifecycleScope.launch {
                                try {
                                    val targetKey = "${model.id}_$selectedResolution"
                                    val targetFile = modelCache.getModelFile(targetKey)
                                    registryClient.downloadModel(model.id, targetFile, selectedResolution, "onnx") { pct, _, _ ->
                                        downloadProgress = pct
                                    }
                                    isDownloaded = true
                                    Toast.makeText(this@MainActivity, "Model (${selectedResolution}x${selectedResolution}) downloaded & verified!", Toast.LENGTH_SHORT).show()
                                } catch (e: Exception) {
                                    Toast.makeText(this@MainActivity, "Download/Export failed: ${e.message}", Toast.LENGTH_LONG).show()
                                } finally {
                                    isDownloading = false
                                }
                            }
                        }
                    },
                    onStartClick = {
                        selectedModel?.let { model ->
                            val targetKey = "${model.id}_$selectedResolution"
                            var file = modelCache.getModelFile(targetKey)
                            if (!file.exists()) {
                                file = modelCache.getModelFile(model.id)
                            }
                            if (file.exists()) {
                                activeDetector = YoloDetector(file, selectedResolution, model.classNames)
                                startCamera(previewView, confThreshold) { res, w, h ->
                                    currentResult = res
                                    frameW = w
                                    frameH = h
                                }
                                isLive = true
                            }
                        }
                    }
                )
            } else {
                LiveDetectionScreen(
                    previewView = previewView,
                    result = currentResult,
                    frameWidth = frameW,
                    frameHeight = frameH,
                    modelTag = "${selectedModel?.name ?: "YOLO"} | $selectedResolution",
                    onStopClick = {
                        stopCamera()
                        activeDetector?.close()
                        activeDetector = null
                        isLive = false
                    }
                )
            }
        }
    }

    private fun fetchModels(serverUrl: String, onResult: (List<RemoteModel>) -> Unit) {
        registryClient.updateServerUrl(serverUrl)
        lifecycleScope.launch {
            try {
                val list = registryClient.getModels()
                onResult(list)
            } catch (e: Exception) {
                Toast.makeText(this@MainActivity, "Could not reach server: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun startCamera(
        previewView: PreviewView,
        confThreshold: Float,
        onResult: (InferenceResult, Int, Int) -> Unit
    ) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also {
                    activeDetector?.let { det ->
                        it.setAnalyzer(cameraExecutor, CameraAnalyzer(det, confThreshold, onResult))
                    }
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)
            } catch (exc: Exception) {
                Toast.makeText(this, "Failed to bind camera: ${exc.message}", Toast.LENGTH_LONG).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun stopCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProviderFuture.get().unbindAll()
        }, ContextCompat.getMainExecutor(this))
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        activeDetector?.close()
    }
}
