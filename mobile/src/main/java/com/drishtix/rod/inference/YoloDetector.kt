package com.drishtix.rod.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.SystemClock
import java.io.File
import java.nio.FloatBuffer
import java.util.Collections
import kotlin.math.max
import kotlin.math.min

class YoloDetector(
    private val modelFile: File,
    val inputResolution: Int = 480,
    private val classNames: List<String>
) {
    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null
    private val distanceEstimator = DistanceEstimator()

    var activeProvider: String = "CPU"
        private set

    // Reusable FloatBuffer for NCHW tensor
    private val inputBuffer: FloatBuffer = FloatBuffer.allocate(1 * 3 * inputResolution * inputResolution)
    private val inputShape = longArrayOf(1, 3, inputResolution.toLong(), inputResolution.toLong())

    // FPS tracking
    private var lastFrameTime = SystemClock.uptimeMillis()
    private val fpsHistory = ArrayDeque<Float>()
    private var currentFps = 0.0f

    init {
        initSession()
    }

    private fun initSession() {
        env = OrtEnvironment.getEnvironment()
        val opts = OrtSession.SessionOptions()
        
        // Attempt NNAPI hardware acceleration
        try {
            opts.addNnapi()
            activeProvider = "NNAPI"
        } catch (e: Exception) {
            activeProvider = "CPU"
        }

        session = env?.createSession(modelFile.absolutePath, opts)
    }

    fun detect(bitmap: Bitmap, confThreshold: Float = 0.25f, iouThreshold: Float = 0.45f): InferenceResult {
        val sess = session ?: return InferenceResult(emptyList(), LatencyMetrics(), 0f, activeProvider)

        val tStart = SystemClock.uptimeMillis()

        // 1. Preprocess: Resize bitmap & normalize to planar FloatBuffer
        val resized = if (bitmap.width != inputResolution || bitmap.height != inputResolution) {
            Bitmap.createScaledBitmap(bitmap, inputResolution, inputResolution, true)
        } else {
            bitmap
        }

        inputBuffer.rewind()
        val pixels = IntArray(inputResolution * inputResolution)
        resized.getPixels(pixels, 0, inputResolution, 0, 0, inputResolution, inputResolution)

        val channelSize = inputResolution * inputResolution
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
        val tPre = SystemClock.uptimeMillis()

        // 2. Inference
        val inputName = sess.inputNames.iterator().next()
        val results = sess.run(Collections.singletonMap(inputName, tensor))
        val tInf = SystemClock.uptimeMillis()

        // 3. Postprocess
        val outputTensor = results[0].value as Array<Array<FloatArray>>
        // Shape: [1][channels][anchors]
        val channels = outputTensor[0].size
        val anchors = outputTensor[0][0].size
        val numClasses = channels - 4

        val candidates = mutableListOf<Detection>()
        val scaleX = bitmap.width.toFloat() / inputResolution
        val scaleY = bitmap.height.toFloat() / inputResolution

        for (a in 0 until anchors) {
            var maxScore = -Float.MAX_VALUE
            var topClass = -1

            for (c in 0 until numClasses) {
                val score = outputTensor[0][4 + c][a]
                if (score > maxScore) {
                    maxScore = score
                    topClass = c
                }
            }

            if (maxScore >= confThreshold) {
                val cx = outputTensor[0][0][a]
                val cy = outputTensor[0][1][a]
                val w = outputTensor[0][2][a]
                val h = outputTensor[0][3][a]

                val x1 = max(0f, (cx - w / 2f) * scaleX)
                val y1 = max(0f, (cy - h / 2f) * scaleY)
                val x2 = min(bitmap.width.toFloat(), (cx + w / 2f) * scaleX)
                val y2 = min(bitmap.height.toFloat(), (cy + h / 2f) * scaleY)

                val rect = RectF(x1, y1, x2, y2)
                val dist = distanceEstimator.estimate(rect, bitmap.width, bitmap.height)
                val cName = if (topClass in classNames.indices) classNames[topClass] else "Class $topClass"

                candidates.add(Detection(rect, maxScore, topClass, cName, dist))
            }
        }

        val filtered = nms(candidates, iouThreshold)
        val tPost = SystemClock.uptimeMillis()

        tensor.close()
        results.close()

        // Calculate metrics
        val latency = LatencyMetrics(
            preprocessMs = tPre - tStart,
            inferenceMs = tInf - tPre,
            postprocessMs = tPost - tInf,
            totalMs = tPost - tStart
        )

        val now = SystemClock.uptimeMillis()
        val delta = now - lastFrameTime
        lastFrameTime = now
        if (delta > 0) {
            fpsHistory.addLast(1000f / delta)
            if (fpsHistory.size > 10) fpsHistory.removeFirst()
            currentFps = fpsHistory.average().toFloat()
        }

        return InferenceResult(filtered, latency, currentFps, activeProvider)
    }

    private fun nms(boxes: List<Detection>, iouThreshold: Float): List<Detection> {
        val sorted = boxes.sortedByDescending { it.score }
        val selected = mutableListOf<Detection>()

        for (candidate in sorted) {
            var keep = true
            for (curr in selected) {
                if (curr.classId == candidate.classId) {
                    if (computeIoU(curr.box, candidate.box) > iouThreshold) {
                        keep = false
                        break
                    }
                }
            }
            if (keep) selected.add(candidate)
        }
        return selected
    }

    private fun computeIoU(a: RectF, b: RectF): Float {
        val left = max(a.left, b.left)
        val top = max(a.top, b.top)
        val right = min(a.right, b.right)
        val bottom = min(a.bottom, b.bottom)

        val interArea = max(0f, right - left) * max(0f, bottom - top)
        if (interArea == 0f) return 0f

        val areaA = a.width() * a.height()
        val areaB = b.width() * b.height()
        return interArea / (areaA + areaB - interArea)
    }

    fun close() {
        session?.close()
        env?.close()
    }
}
