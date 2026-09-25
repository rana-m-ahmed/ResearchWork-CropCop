package org.cropcop.trackc

import android.app.Activity
import android.os.Bundle
import android.os.Process
import android.widget.TextView
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

        if (mode == "raw_fidelity") error("TRACKC_RAW_PIPELINE_NOT_IMPLEMENTED_EXACT_CTC_V2")
        val tensors = verifiedTensors()
        fun forward(index: Int): FloatArray {
            val value = Tensor.fromBlob(readTensor(tensors[index]), longArrayOf(1, 3, 256, 256))
            return module.forward(EValue.from(value))?.firstOrNull()?.toTensor()?.dataAsFloatArray
                ?: error("TRACKC_FORWARD_EMPTY")
        }
        fun top1(values: FloatArray): Int = values.indices.maxBy { values[it] }
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
            "end_to_end" -> runTimed("end_to_end", 768, true)
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
}
