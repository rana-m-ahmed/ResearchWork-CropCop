package org.cropcop.trackc

import android.app.Activity
import android.os.Bundle
import android.os.Process
import android.widget.TextView
import org.json.JSONObject
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
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

        if (mode in setOf("tensor_fidelity", "raw_fidelity", "model_latency", "end_to_end", "warmup_transition")) {
            error("TRACKC_INPUT_MANIFEST_NOT_MATERIALIZED")
        }

        // This is unreachable for current frozen modes, but retains the exact forward boundary.
        val tensor = Tensor.fromBlob(FloatArray(1 * 3 * 256 * 256), longArrayOf(1, 3, 256, 256))
        val logits = module.forward(EValue.from(tensor))?.firstOrNull()?.toTensor()?.dataAsFloatArray
            ?: error("TRACKC_FORWARD_EMPTY")
        require(logits.size == 120) { "TRACKC_OUTPUT_WIDTH_MISMATCH" }
        return report.put("output_width", logits.size)
    }
}
