package org.cropcop.trackc

import android.app.Activity
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.os.Bundle
import android.os.Process
import android.widget.TextView
import androidx.exifinterface.media.ExifInterface
import org.json.JSONObject
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.io.File

class BenchmarkActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(TextView(this).apply { text = "RUNNING_NONCLAIM" })
        Thread {
            val report = runCatching {
                execute(
                    intent.getStringExtra("trackc_mode") ?: "system_snapshot",
                    intent.getStringExtra("trackc_variant") ?: "int8",
                )
            }.fold(
                onSuccess = { it.put("status", "READY_NONCLAIM") },
                onFailure = { JSONObject().put("status", "BLOCKED").put("reason", it.message) },
            )
            runOnUiThread { setContentView(TextView(this).apply { text = report.toString() }) }
        }.start()
    }

    private fun execute(requestedMode: String, requestedVariant: String): JSONObject {
        val mode = BenchmarkContract.requireMode(requestedMode)
        val artifactSpec = BenchmarkContract.artifact(requestedVariant)
        val report = JSONObject()
            .put("mode", mode)
            .put("pid", Process.myPid())
            .put("abi", android.os.Build.SUPPORTED_ABIS.joinToString(","))
            .put("runtime", "ExecuTorch ${BenchmarkContract.executorchVersion}")
            .put("requested_threads", BenchmarkContract.requestedThreads)

        if (mode == "system_snapshot") return report

        val artifact = File(filesDir, "trackc/${artifactSpec.filename}")
        BenchmarkContract.verifyArtifact(artifact, artifactSpec)
        val module = Module.load(
            artifact.absolutePath,
            Module.LOAD_MODE_FILE,
            BenchmarkContract.requestedThreads,
        ) ?: error("TRACKC_MODULE_LOAD_NULL")

        if (mode == "load") return report.put("artifact_variant", artifactSpec.id).put("artifact_sha256", artifactSpec.sha256)

        val tensors = verifiedTensors()
        fun forwardValue(value: Tensor): FloatArray = module.forward(EValue.from(value))?.firstOrNull()?.toTensor()?.dataAsFloatArray
            ?: error("TRACKC_FORWARD_EMPTY")
        fun forward(index: Int): FloatArray {
            val value = Tensor.fromBlob(readTensor(tensors[index]), longArrayOf(1, 3, 256, 256))
            return forwardValue(value)
        }
        fun top1(values: FloatArray): Int = values.indices.maxBy { values[it] }
        fun rawForward(index: Int): FloatArray {
            val value = Tensor.fromBlob(preprocessRaw(verifiedRawFiles()[index]), longArrayOf(1, 3, 256, 256))
            return module.forward(EValue.from(value))?.firstOrNull()?.toTensor()?.dataAsFloatArray ?: error("TRACKC_FORWARD_EMPTY")
        }
        val evidence = File(filesDir, "trackc/evidence").apply { mkdirs() }
        fun evidenceName(name: String): String = if (artifactSpec.id == "int8") name else "${name}_${artifactSpec.id}"
        fun runTimed(name: String, count: Int, inputFor: (Int) -> Tensor): JSONObject {
            val output = File(evidence, "${evidenceName(name)}.csv")
            output.printWriter().use { writer ->
                writer.println("sample_index,elapsed_ns,top1")
                repeat(count) { i ->
                    val input = inputFor(i)
                    val start = System.nanoTime()
                    val logits = forwardValue(input)
                    val elapsed = System.nanoTime() - start
                    require(logits.size == 120) { "TRACKC_OUTPUT_WIDTH_MISMATCH" }
                    writer.println("${i % tensors.size},$elapsed,${top1(logits)}")
                }
            }
            return JSONObject().put("samples", count).put("evidence", output.name).put("evidence_sha256", BenchmarkContract.sha256(output))
        }

        return when (mode) {
            "tensor_fidelity" -> runTimed("tensor_fidelity", 256) { i -> Tensor.fromBlob(readTensor(tensors[i]), longArrayOf(1, 3, 256, 256)) }
                .put("input_manifest_sha256", BenchmarkContract.tensorManifestSha256)
            "warmup_transition" -> {
                val input = Tensor.fromBlob(readTensor(tensors[0]), longArrayOf(1, 3, 256, 256))
                runTimed("warmup_transition", 10) { input }
            }
            "model_latency" -> {
                val input = Tensor.fromBlob(readTensor(tensors[0]), longArrayOf(1, 3, 256, 256))
                repeat(50) { forwardValue(input) }
                runTimed("model_latency", 1000) { input }.put("untimed_warmups", 50)
            }
            "end_to_end" -> {
                val output = File(evidence, "${evidenceName("end_to_end")}.csv")
                output.printWriter().use { writer ->
                    writer.println("sample_index,elapsed_ns,top1")
                    repeat(768) { i ->
                        val start = System.nanoTime()
                        val logits = rawForward(i % tensors.size)
                        val elapsed = System.nanoTime() - start
                        require(logits.size == 120) { "TRACKC_OUTPUT_WIDTH_MISMATCH" }
                        writer.println("${i % tensors.size},$elapsed,${top1(logits)}")
                    }
                }
                JSONObject().put("samples", 768).put("evidence", output.name).put("evidence_sha256", BenchmarkContract.sha256(output)).put("boundary", "raw_decode_preprocess_forward")
            }
            "raw_fidelity" -> {
                val output = File(evidence, "${evidenceName("raw_fidelity")}.csv")
                var matches = 0
                output.printWriter().use { writer ->
                    writer.println("sample_index,raw_top1,tensor_top1,match")
                    repeat(256) { i ->
                        val raw = top1(rawForward(i)); val tensor = top1(forward(i))
                        if (raw == tensor) matches++
                        writer.println("$i,$raw,$tensor,${raw == tensor}")
                    }
                }
                JSONObject().put("samples", 256).put("matches", matches).put("agreement", matches / 256.0)
                    .put("evidence", output.name).put("evidence_sha256", BenchmarkContract.sha256(output))
            }
            else -> error("TRACKC_MODE_DISPATCH_ERROR")
        }.also { report.put("result", it) }
    }

    private fun verifiedTensors(): List<File> {
        val root = File(filesDir, "trackc/inputs")
        val manifest = File(root, "manifests/DS_DEVICE_TENSOR_256_R07_manifest.csv")
        require(BenchmarkContract.sha256(manifest) == BenchmarkContract.tensorManifestSha256) { "TRACKC_TENSOR_MANIFEST_SHA256_MISMATCH" }
        require(BenchmarkContract.sha256(File(root, "locks/TRACKC_R07_DEVICE_INPUT_LOCK_v1.json")) == BenchmarkContract.inputLockSha256) { "TRACKC_INPUT_LOCK_SHA256_MISMATCH" }
        val tensors = File(root, "tensor_256").listFiles { file -> file.name.endsWith(".f32le.bin") }?.sortedBy { it.name }
            ?: error("TRACKC_TENSOR_DIRECTORY_UNAVAILABLE")
        require(tensors.size == 256) { "TRACKC_TENSOR_COUNT_MISMATCH:${tensors.size}" }
        require(tensors.all { it.length() == 786432L }) { "TRACKC_TENSOR_BYTES_MISMATCH" }
        return tensors
    }

    private fun readTensor(file: File): FloatArray {
        val bytes = file.readBytes()
        require(bytes.size == 786432) { "TRACKC_TENSOR_BYTES_MISMATCH" }
        return FloatArray(196608).also { ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(it) }
    }

    private fun verifiedRawFiles(): List<File> {
        val root = File(filesDir, "trackc/inputs")
        val manifest = File(root, "manifests/DS_DEVICE_RAW_256_R07_manifest.csv")
        require(BenchmarkContract.sha256(manifest) == BenchmarkContract.rawManifestSha256) { "TRACKC_RAW_MANIFEST_SHA256_MISMATCH" }
        val lines = manifest.readLines().drop(1)
        require(lines.size == 256) { "TRACKC_RAW_COUNT_MISMATCH" }
        return lines.map { line -> File(root, line.split(',')[6]) }.also { files -> require(files.all { it.isFile }) { "TRACKC_RAW_FILE_UNAVAILABLE" } }
    }

    private fun preprocessRaw(file: File): FloatArray {
        var source = BitmapFactory.decodeFile(file.absolutePath) ?: error("TRACKC_RAW_DECODE_FAILED")
        val orientation = ExifInterface(file.absolutePath).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
        val matrix = Matrix().apply { when (orientation) { ExifInterface.ORIENTATION_ROTATE_90 -> postRotate(90f); ExifInterface.ORIENTATION_ROTATE_180 -> postRotate(180f); ExifInterface.ORIENTATION_ROTATE_270 -> postRotate(270f) } }
        if (!matrix.isIdentity) source = Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
        val pixels = IntArray(source.width * source.height); source.getPixels(pixels, 0, source.width, 0, 0, source.width, source.height)
        val mean = IntArray(3); pixels.forEach { p -> mean[0] += Color.red(p); mean[1] += Color.green(p); mean[2] += Color.blue(p) }; mean.indices.forEach { mean[it] = kotlin.math.round(mean[it].toDouble() / pixels.size).toInt() }
        val side = maxOf(source.width, source.height); val square = Bitmap.createBitmap(side, side, Bitmap.Config.ARGB_8888)
        Canvas(square).apply { drawColor(Color.rgb(mean[0], mean[1], mean[2])); drawBitmap(source, ((side-source.width)/2).toFloat(), ((side-source.height)/2).toFloat(), null) }
        return resizeBicubicAntialias(square)
    }

    // Separable Keys cubic (a=-0.5), with a scale-aware support window and
    // clamped-edge normalization. This is the deterministic CTC-v2 bicubic
    // antialias contract, without Android's bilinear Bitmap scaler.
    private fun resizeBicubicAntialias(source: Bitmap): FloatArray {
        val width = source.width; val height = source.height; val pixels = IntArray(width * height)
        source.getPixels(pixels, 0, width, 0, 0, width, height)
        data class Contribution(val indices: IntArray, val weights: FloatArray)
        fun kernel(x: Double): Double { val a = -0.5; val t = kotlin.math.abs(x); return when { t < 1 -> (a + 2)*t*t*t - (a + 3)*t*t + 1; t < 2 -> a*t*t*t - 5*a*t*t + 8*a*t - 4*a; else -> 0.0 } }
        fun contributions(inputSize: Int): Array<Contribution> {
            val scale = inputSize / 256.0
            val filterScale = maxOf(1.0, scale)
            val support = 2.0 * filterScale
            return Array(256) { destination ->
                val center = (destination + 0.5) * scale
                val left = kotlin.math.ceil(center - support).toInt()
                val right = kotlin.math.floor(center + support).toInt()
                val indices = IntArray(right - left + 1)
                val weights = FloatArray(indices.size)
                var total = 0.0
                for (offset in indices.indices) {
                    val sourceIndex = left + offset
                    indices[offset] = sourceIndex.coerceIn(0, inputSize - 1)
                    val weight = kernel((sourceIndex - center + 0.5) / filterScale)
                    weights[offset] = weight.toFloat()
                    total += weight
                }
                for (offset in weights.indices) weights[offset] = (weights[offset] / total).toFloat()
                Contribution(indices, weights)
            }
        }
        val horizontal = contributions(width); val vertical = contributions(height)
        val intermediate = FloatArray(height * 256 * 3)
        for (y in 0 until height) for (x in 0 until 256) {
            val contribution = horizontal[x]
            for (c in 0..2) {
                var sum = 0f
                for (k in contribution.indices.indices) {
                    val pixel = pixels[y * width + contribution.indices[k]]
                    val channel = when (c) { 0 -> Color.red(pixel); 1 -> Color.green(pixel); else -> Color.blue(pixel) }
                    sum += channel * contribution.weights[k]
                }
                intermediate[(y * 256 + x) * 3 + c] = sum
            }
        }
        return FloatArray(3 * 256 * 256).also { out ->
            for (y in 0 until 256) for (x in 0 until 256) for (c in 0..2) {
                val contribution = vertical[y]; var sum = 0f
                for (k in contribution.indices.indices) sum += intermediate[(contribution.indices[k] * 256 + x) * 3 + c] * contribution.weights[k]
                // TorchVision resizes the PIL image before pil_to_tensor, so
                // the resampled channel is quantized to uint8 at this boundary.
                val v = kotlin.math.round(sum).toInt().coerceIn(0, 255) / 255f; val i = y * 256 + x + c * 65536
                out[i] = when (c) { 0 -> (v - .485f) / .229f; 1 -> (v - .456f) / .224f; else -> (v - .406f) / .225f }
            }
        }
    }
}
