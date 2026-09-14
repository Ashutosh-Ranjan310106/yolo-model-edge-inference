package com.drishtix.rod.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.os.SystemClock
import java.io.File
import java.nio.FloatBuffer
import java.util.Collections
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

data class DepthSectorInfo(
    val medianDepth: Float,
    val minDepth: Float,
    val isObstacle: Boolean,
    val label: String
)

data class DepthResult(
    val depthMap: Array<FloatArray>, // [H][W] depth values
    val resolution: Int,
    val latencyMs: Long,
    val provider: String,
    val sectors: Map<String, DepthSectorInfo>,
    val letterboxInfo: LetterboxInfo
)

data class LetterboxInfo(
    val scale: Float,
    val padX: Int,
    val padY: Int,
    val newW: Int,
    val newH: Int,
    val targetSize: Int,
    val origWidth: Int,
    val origHeight: Int
)

/**
 * Mobile On-Device Depth Estimator using ONNX Runtime.
 * Supports YOLO26-Depth at 512x512 (Quality mode, default) and 320x320 (Speed/Battery mode).
 * Uses neutral gray (114, 114, 114) letterboxing to preserve camera native 16:9 aspect ratio.
 */
class DepthEstimator(
    private val modelFile: File,
    val resolution: Int = 512,
    val isMetric: Boolean = true
) {
    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null

    var activeProvider: String = "CPU"
        private set

    var droppedFrames: Long = 0
        private set

    var lastLatencyMs: Long = 0
        private set

    // Reusable FloatBuffer for NCHW [1, 3, H, W]
    private val inputBuffer: FloatBuffer = FloatBuffer.allocate(1 * 3 * resolution * resolution)
    private val inputShape = longArrayOf(1, 3, resolution.toLong(), resolution.toLong())

    // Letterbox working canvas & bitmap
    private val letterboxBitmap: Bitmap = Bitmap.createBitmap(resolution, resolution, Bitmap.Config.ARGB_8888)
    private val letterboxCanvas: Canvas = Canvas(letterboxBitmap)
    private val neutralGrayPaint = Paint().apply { color = Color.rgb(114, 114, 114) }

    init {
        initSession()
    }

    private fun initSession() {
        env = OrtEnvironment.getEnvironment()
        val opts = OrtSession.SessionOptions()

        try {
            opts.addNnapi()
            activeProvider = "NNAPI"
        } catch (e: Exception) {
            activeProvider = "CPU"
        }

        session = env?.createSession(modelFile.absolutePath, opts)
    }

    /**
     * Compute aspect-ratio-preserving letterbox scale and padding.
     */
    fun computeLetterbox(srcW: Int, srcH: Int): LetterboxInfo {
        val scale = min(resolution.toFloat() / srcW, resolution.toFloat() / srcH)
        val newW = (srcW * scale).roundToInt()
        val newH = (srcH * scale).roundToInt()
        val padX = (resolution - newW) / 2
        val padY = (resolution - newH) / 2

        return LetterboxInfo(
            scale = scale,
            padX = padX,
            padY = padY,
            newW = newW,
            newH = newH,
            targetSize = resolution,
            origWidth = srcW,
            origHeight = srcH
        )
    }

    /**
     * Run depth estimation on an input camera frame.
     */
    fun estimate(srcBitmap: Bitmap): DepthResult? {
        val sess = session ?: return null
        val tStart = SystemClock.uptimeMillis()

        val lb = computeLetterbox(srcBitmap.width, srcBitmap.height)

        // 1. Draw neutral gray (114, 114, 114) background
        letterboxCanvas.drawRect(0f, 0f, resolution.toFloat(), resolution.toFloat(), neutralGrayPaint)

        // 2. Draw scaled camera frame centered
        val dstRect = RectF(
            lb.padX.toFloat(),
            lb.padY.toFloat(),
            (lb.padX + lb.newW).toFloat(),
            (lb.padY + lb.newH).toFloat()
        )
        letterboxCanvas.drawBitmap(srcBitmap, null, dstRect, null)

        // 3. Preprocess to FloatBuffer (0.0 to 1.0 normalization)
        inputBuffer.rewind()
        val pixels = IntArray(resolution * resolution)
        letterboxBitmap.getPixels(pixels, 0, resolution, 0, 0, resolution, resolution)

        val channelSize = resolution * resolution
        val rArr = FloatArray(channelSize)
        val gArr = FloatArray(channelSize)
        val bArr = FloatArray(channelSize)

        for (i in pixels.indices) {
            val p = pixels[i]
            rArr[i] = ((p shr 16) and 0xFF) / 255.0f
            gArr[i] = ((p shr 8) and 0xFF) / 255.0f
            bArr[i] = (p and 0xFF) / 255.0f
        }

        inputBuffer.put(rArr)
        inputBuffer.put(gArr)
        inputBuffer.put(bArr)
        inputBuffer.rewind()

        val tensor = OnnxTensor.createTensor(env, inputBuffer, inputShape)

        // 4. Run ONNX Inference
        val inputName = sess.inputNames.iterator().next()
        val results = sess.run(Collections.singletonMap(inputName, tensor))
        val tInf = SystemClock.uptimeMillis()
        lastLatencyMs = tInf - tStart

        // 5. Extract output tensor [1, 1, resolution, resolution]
        val rawOutput = results[0].value as Array<Array<Array<FloatArray>>>
        val depth2D = rawOutput[0][0] // [H][W]

        // 6. Compute 3-sector hazards
        val sectors = computeSectors(depth2D, lb)

        return DepthResult(
            depthMap = depth2D,
            resolution = resolution,
            latencyMs = lastLatencyMs,
            provider = activeProvider,
            sectors = sectors,
            letterboxInfo = lb
        )
    }

    /**
     * Extracts median depth within an object's bounding box in original camera coordinates.
     */
    fun getMedianDepthInROI(
        depthMap: Array<FloatArray>,
        lb: LetterboxInfo,
        box: RectF
    ): Float? {
        // Map original camera pixel ROI into active letterbox region
        val tensorX1 = ((box.left * lb.scale) + lb.padX).toInt().coerceIn(lb.padX, lb.padX + lb.newW - 1)
        val tensorY1 = ((box.top * lb.scale) + lb.padY).toInt().coerceIn(lb.padY, lb.padY + lb.newH - 1)
        val tensorX2 = ((box.right * lb.scale) + lb.padX).toInt().coerceIn(lb.padX, lb.padX + lb.newW - 1)
        val tensorY2 = ((box.bottom * lb.scale) + lb.padY).toInt().coerceIn(lb.padY, lb.padY + lb.newH - 1)

        val minX = min(tensorX1, tensorX2)
        val maxX = max(tensorX1, tensorX2)
        val minY = min(tensorY1, tensorY2)
        val maxY = max(tensorY1, tensorY2)

        val sampleValues = mutableListOf<Float>()
        val stepX = max(1, (maxX - minX) / 12)
        val stepY = max(1, (maxY - minY) / 12)

        for (y in minY..maxY step stepY) {
            for (x in minX..maxX step stepX) {
                sampleValues.add(depthMap[y][x])
            }
        }

        if (sampleValues.isEmpty()) return null
        sampleValues.sort()
        return sampleValues[sampleValues.size / 2]
    }

    private fun computeSectors(
        depthMap: Array<FloatArray>,
        lb: LetterboxInfo
    ): Map<String, DepthSectorInfo> {
        val yStart = lb.padY + (lb.newH * 0.35f).toInt()
        val yEnd = lb.padY + (lb.newH * 0.85f).toInt()
        val sectorW = lb.newW / 3

        val leftSamples = mutableListOf<Float>()
        val centerSamples = mutableListOf<Float>()
        val rightSamples = mutableListOf<Float>()

        val stepY = max(1, (yEnd - yStart) / 10)
        val stepX = max(1, sectorW / 10)

        for (y in yStart until yEnd step stepY) {
            // Left sector
            val lxStart = lb.padX
            val lxEnd = lb.padX + sectorW
            for (x in lxStart until lxEnd step stepX) {
                leftSamples.add(depthMap[y][x])
            }

            // Center sector
            val cxStart = lb.padX + sectorW
            val cxEnd = lb.padX + 2 * sectorW
            for (x in cxStart until cxEnd step stepX) {
                centerSamples.add(depthMap[y][x])
            }

            // Right sector
            val rxStart = lb.padX + 2 * sectorW
            val rxEnd = lb.padX + lb.newW
            for (x in rxStart until rxEnd step stepX) {
                rightSamples.add(depthMap[y][x])
            }
        }

        fun analyze(samples: MutableList<Float>): DepthSectorInfo {
            if (samples.isEmpty()) return DepthSectorInfo(10f, 10f, false, "CLEAR")
            samples.sort()
            val med = samples[samples.size / 2]
            val minD = samples[0]
            val isObs = med <= 2.0f
            val label = if (med <= 1.2f) "BLOCKED (<1.2m)" else if (isObs) "CAUTION (${String.format("%.1f", med)}m)" else "CLEAR"
            return DepthSectorInfo(med, minD, isObs, label)
        }

        return mapOf(
            "left" to analyze(leftSamples),
            "center" to analyze(centerSamples),
            "right" to analyze(rightSamples)
        )
    }

    fun recordDroppedFrame() {
        droppedFrames++
    }

    fun close() {
        session?.close()
        session = null
    }
}
