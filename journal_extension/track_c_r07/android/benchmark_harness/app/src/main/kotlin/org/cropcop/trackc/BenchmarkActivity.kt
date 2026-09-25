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
        val report = runCatching { execute(intent.getStringExtra("trackc_mode") ?: "system_snapshot") }
            .fold(
                onSuccess = { it.put("status", "READY_NONCLAIM") },
                onFailure = { JSONObject().put("status", "BLOCKED").put("reason", it.message) },
            )
        setContentView(TextView(this).apply { text = report.toString() })
    }

    private fun execute(requestedMode: String): JSONObject {
        val mode = BenchmarkContract.requireMode(requestedMode)
        val report = JSONObject()
            .put("mode", mode)
            .put("pid", Process.myPid())
            .put("abi", android.os.Build.SUPPORTED_ABIS.joinToString(","))
            .put("runtime", "ExecuTorch ${BenchmarkContract.executorchVersion}")
            .put("requested_threads", BenchmarkContract.requestedThreads)

        if (mode == "system_snapshot") return report

        val artifact = File(filesDir, "trackc/r07_s1_xnnpack_int8.pte")
        BenchmarkContract.verifyArtifact(artifact)
        val module = Module.load(
            artifact.absolutePath,
            Module.LOAD_MODE_FILE,
            BenchmarkContract.requestedThreads,
        ) ?: error("TRACKC_MODULE_LOAD_NULL")

        if (mode == "load") return report.put("artifact_sha256", BenchmarkContract.artifactSha256)

        val tensors = verifiedTensors()
        fun forward(index: Int): FloatArray {
            val value = Tensor.fromBlob(readTensor(tensors[index]), longArrayOf(1, 3, 256, 256))
            return module.forward(EValue.from(value))?.firstOrNull()?.toTensor()?.dataAsFloatArray
                ?: error("TRACKC_FORWARD_EMPTY")
        }
        fun top1(values: FloatArray): Int = values.indices.maxBy { values[it] }
        fun rawForward(index: Int): FloatArray {
            val value = Tensor.fromBlob(preprocessRaw(verifiedRawFiles()[index]), longArrayOf(1, 3, 256, 256))
            return module.forward(EValue.from(value))?.firstOrNull()?.toTensor()?.dataAsFloatArray ?: error("TRACKC_FORWARD_EMPTY")
        }
        val evidence = File(filesDir, "trackc/evidence").apply { mkdirs() }
        fun runTimed(name: String, count: Int, includeRead: Boolean): JSONObject {
            val output = File(evidence, "$name.csv")
            output.printWriter().use { writer ->
                writer.println("sample_index,elapsed_ns,top1")
                repeat(count) { i ->
                    val start = System.nanoTime()
                    val logits = forward(i % tensors.size)
                    val elapsed = System.nanoTime() - start
                    require(logits.size == 120) { "TRACKC_OUTPUT_WIDTH_MISMATCH" }
                    writer.println("${i % tensors.size},$elapsed,${top1(logits)}")
                }
            }
            return JSONObject().put("samples", count).put("evidence", output.name).put("evidence_sha256", BenchmarkContract.sha256(output))
        }

        return when (mode) {
            "tensor_fidelity" -> runTimed("tensor_fidelity", 256, true)
                .put("input_manifest_sha256", BenchmarkContract.tensorManifestSha256)
            "warmup_transition" -> runTimed("warmup_transition", 10, false)
            "model_latency" -> {
                repeat(50) { forward(it % tensors.size) }
                runTimed("model_latency", 1000, false).put("untimed_warmups", 50)
            }
            "end_to_end" -> {
                val output = File(evidence, "end_to_end.csv")
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
                val output = File(evidence, "raw_fidelity.csv")
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

    // This bounded cubic implementation replaces Android's bilinear scaler.
    // Its raw-vs-canonical agreement is measured, not assumed equivalent.
    private fun resizeBicubicAntialias(source: Bitmap): FloatArray {
        val width = source.width; val height = source.height; val pixels = IntArray(width * height)
        source.getPixels(pixels, 0, width, 0, 0, width, height)
        val sx = width / 256.0; val sy = height / 256.0
        fun kernel(x: Double): Double { val a = -0.5; val t = kotlin.math.abs(x); return when { t < 1 -> (a + 2)*t*t*t - (a + 3)*t*t + 1; t < 2 -> a*t*t*t - 5*a*t*t + 8*a*t - 4*a; else -> 0.0 } }
        fun sample(x: Int, y: Int, channel: Int): Int { val p = pixels[y.coerceIn(0,height-1)*width+x.coerceIn(0,width-1)]; return when(channel){0->Color.red(p);1->Color.green(p);else->Color.blue(p)} }
        return FloatArray(3 * 256 * 256).also { out ->
            for (y in 0 until 256) for (x in 0 until 256) for (c in 0..2) {
                val cx=(x+.5)*sx-.5; val cy=(y+.5)*sy-.5; val rx=2.0; val ry=2.0
                var sum=0.0; var weight=0.0
                for (iy in kotlin.math.floor(cy-ry).toInt()..kotlin.math.ceil(cy+ry).toInt()) for (ix in kotlin.math.floor(cx-rx).toInt()..kotlin.math.ceil(cx+rx).toInt()) {
                    val wx=kernel(cx-ix); val wy=kernel(cy-iy); val w=wx*wy; sum+=w*sample(ix,iy,c); weight+=w
                }
                val v=(sum/weight/255.0).toFloat(); val i=y*256+x+c*65536
                out[i]=when(c){0->(v-.485f)/.229f;1->(v-.456f)/.224f;else->(v-.406f)/.225f}
            }
        }
    }
}
