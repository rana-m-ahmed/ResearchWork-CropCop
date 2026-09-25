package org.cropcop.trackc

import java.io.File
import java.security.MessageDigest

object BenchmarkContract {
    const val artifactSha256 = "2e0c54a1b5bb7c0018d0159a642e49e4f4bd79fcc1f940e215a40734a2695bb8"
    const val artifactBytes = 28_555_872L
    const val executorchVersion = "1.3.1"
    const val requestedThreads = 4
    val modes = setOf(
        "tensor_fidelity",
        "raw_fidelity",
        "load",
        "warmup_transition",
        "model_latency",
        "end_to_end",
        "system_snapshot",
    )

    fun requireMode(value: String): String =
        value.also { require(it in modes) { "TRACKC_UNKNOWN_MODE:$it" } }

    fun verifyArtifact(file: File) {
        require(file.isFile) { "TRACKC_ARTIFACT_UNAVAILABLE:${file.absolutePath}" }
        require(file.length() == artifactBytes) { "TRACKC_ARTIFACT_BYTES_MISMATCH" }
        require(sha256(file) == artifactSha256) { "TRACKC_ARTIFACT_SHA256_MISMATCH" }
    }

    fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().buffered().use { input ->
            val buffer = ByteArray(1024 * 1024)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                if (count > 0) digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}
