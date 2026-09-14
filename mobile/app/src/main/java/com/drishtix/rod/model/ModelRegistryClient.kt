package com.drishtix.rod.model

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.TimeUnit

data class RemoteModel(
    val id: String = "",
    val name: String,
    val category: String = "training_checkpoint",
    val runId: String? = null,
    val epoch: Int? = null,
    val variant: String = "n",
    val resolution: Int = 480,
    val format: String = "pt",
    val fileSizeMb: Float = 0.0f,
    val version: String = "1.0",
    val sha256: String = "",
    val downloadUrl: String,
    val supportedResolutions: List<Int> = listOf(320, 480, 640),
    val classNames: List<String> = emptyList()
)

class ModelRegistryClient(private var serverBaseUrl: String) {

    private val client = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .build()

    fun updateServerUrl(url: String) {
        serverBaseUrl = url.trimEnd('/')
    }

    suspend fun getModels(): List<RemoteModel> = withContext(Dispatchers.IO) {
        val request = Request.Builder()
            .url("$serverBaseUrl/api/models")
            .build()

        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) throw Exception("Failed to fetch models: HTTP ${response.code}")
            val body = response.body?.string() ?: throw Exception("Empty response from server")
            val json = JSONObject(body)
            val array = json.getJSONArray("models")

            val list = mutableListOf<RemoteModel>()
            for (i in 0 until array.length()) {
                val item = array.getJSONObject(i)
                val classNamesList = mutableListOf<String>()
                if (item.has("class_names")) {
                    val cArr = item.getJSONArray("class_names")
                    for (c in 0 until cArr.length()) {
                        classNamesList.add(cArr.getString(c))
                    }
                }
                val resolutionsList = mutableListOf<Int>()
                if (item.has("supported_resolutions")) {
                    val rArr = item.getJSONArray("supported_resolutions")
                    for (r in 0 until rArr.length()) {
                        resolutionsList.add(rArr.getInt(r))
                    }
                }
                list.add(
                    RemoteModel(
                        id = item.getString("id"),
                        name = item.getString("name"),
                        category = item.optString("category", "training_checkpoint"),
                        runId = if (item.has("run_id") && !item.isNull("run_id")) item.getString("run_id") else null,
                        epoch = if (item.has("epoch") && !item.isNull("epoch")) item.getInt("epoch") else null,
                        variant = item.optString("variant", "n"),
                        resolution = item.optInt("resolution", 480),
                        format = item.optString("format", "pt"),
                        fileSizeMb = item.optDouble("file_size_mb", 0.0).toFloat(),
                        version = item.optString("version", "1.0"),
                        sha256 = item.optString("sha256", ""),
                        downloadUrl = item.getString("download_url"),
                        supportedResolutions = if (resolutionsList.isNotEmpty()) resolutionsList else listOf(320, 480, 640),
                        classNames = classNamesList
                    )
                )
            }
            list
        }
    }

    suspend fun downloadModel(
        modelId: String,
        targetFile: File,
        resolution: Int = 480,
        format: String = "onnx",
        onProgress: (percent: Int, bytesRead: Long, totalBytes: Long) -> Unit
    ) = withContext(Dispatchers.IO) {
        val url = "$serverBaseUrl/api/models/$modelId/download?resolution=$resolution&format=$format"
        val request = Request.Builder().url(url).build()

        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) throw Exception("Download failed: HTTP ${response.code}")
            val body = response.body ?: throw Exception("Empty download response body")
            val contentLength = body.contentLength()

            val tempFile = File(targetFile.parentFile, "${targetFile.name}.download")
            tempFile.parentFile?.mkdirs()

            body.byteStream().use { input ->
                FileOutputStream(tempFile).use { output ->
                    val buffer = ByteArray(32 * 1024)
                    var bytesRead = 0L
                    var read: Int

                    while (input.read(buffer).also { read = it } != -1) {
                        output.write(buffer, 0, read)
                        bytesRead += read
                        val percent = if (contentLength > 0) ((bytesRead * 100) / contentLength).toInt() else 0
                        onProgress(percent, bytesRead, contentLength)
                    }
                    output.flush()
                }
            }

            if (tempFile.exists()) {
                if (targetFile.exists()) targetFile.delete()
                tempFile.renameTo(targetFile)
            }
        }
    }
}
