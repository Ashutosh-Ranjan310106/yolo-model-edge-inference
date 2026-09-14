package com.drishtix.rod.model

import android.content.Context
import java.io.File
import java.io.FileInputStream
import java.security.MessageDigest

class LocalModelCache(private val context: Context) {

    private val modelsDir: File = File(context.filesDir, "models").apply {
        if (!exists()) mkdirs()
    }

    fun getModelFile(modelId: String): File {
        return File(modelsDir, "$modelId.onnx")
    }

    fun isModelDownloaded(modelId: String): Boolean {
        val file = getModelFile(modelId)
        return file.exists() && file.length() > 0
    }

    fun deleteModel(modelId: String): Boolean {
        val file = getModelFile(modelId)
        return if (file.exists()) file.delete() else false
    }

    fun calculateSha256(file: File): String {
        if (!file.exists()) return ""
        val digest = MessageDigest.getInstance("SHA-256")
        FileInputStream(file).use { fis ->
            val buffer = ByteArray(8192)
            var read: Int
            while (fis.read(buffer).also { read = it } != -1) {
                digest.update(buffer, 0, read)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}
