package com.drishtix.rod.camera

import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.drishtix.rod.inference.DepthEstimator
import com.drishtix.rod.inference.DepthResult
import com.drishtix.rod.inference.InferenceResult
import com.drishtix.rod.inference.YoloDetector
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class CameraAnalyzer(
    private val detector: YoloDetector,
    private val confThreshold: Float,
    private val depthEstimator: DepthEstimator? = null,
    private val onDepthResult: ((DepthResult) -> Unit)? = null,
    private val onResult: (InferenceResult, Int, Int) -> Unit
) : ImageAnalysis.Analyzer {

    private val isBusy = AtomicBoolean(false)
    private val isDepthBusy = AtomicBoolean(false)
    private val depthExecutor = Executors.newSingleThreadExecutor()

    override fun analyze(imageProxy: ImageProxy) {
        // Drop frame immediately if YOLO inference is still busy (latest-frame strategy)
        if (!isBusy.compareAndSet(false, true)) {
            imageProxy.close()
            return
        }

        try {
            val bitmap = imageProxy.toBitmap()
            val rotationDegrees = imageProxy.imageInfo.rotationDegrees

            val rotated = if (rotationDegrees != 0) {
                val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
                Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            } else {
                bitmap
            }

            // Run depth asynchronously and independently if worker is not busy
            if (depthEstimator != null && onDepthResult != null) {
                if (isDepthBusy.compareAndSet(false, true)) {
                    val frameForDepth = rotated.copy(rotated.config ?: Bitmap.Config.ARGB_8888, false)
                    depthExecutor.execute {
                        try {
                            val depthRes = depthEstimator.estimate(frameForDepth)
                            if (depthRes != null) {
                                onDepthResult.invoke(depthRes)
                            }
                        } catch (e: Exception) {
                            e.printStackTrace()
                        } finally {
                            isDepthBusy.set(false)
                        }
                    }
                } else {
                    depthEstimator.recordDroppedFrame()
                }
            }

            val result = detector.detect(rotated, confThreshold)
            onResult(result, rotated.width, rotated.height)
        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            imageProxy.close()
            isBusy.set(false)
        }
    }
}
