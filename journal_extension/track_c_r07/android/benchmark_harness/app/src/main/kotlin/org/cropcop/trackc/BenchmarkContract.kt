package org.cropcop.trackc

import java.io.File
import java.security.MessageDigest

object BenchmarkContract {
    data class ArtifactSpec(val id: String, val filename: String, val sha256: String, val bytes: Long)
    const val artifactSha256 = "2e0c54a1b5bb7c0018d0159a642e49e4f4bd79fcc1f940e215a40734a2695bb8"
    const val artifactBytes = 28_555_872L
    const val fp32ArtifactSha256 = "61556330cd9fc4725b4ff4759aef3835696b81865b2cbe030f815a4c0e8d5aec"
    const val fp32ArtifactBytes = 111_741_536L
    val executorchVersion: String
        get() = BuildConfig.EXECUTORCH_VERSION
    const val requestedThreads = 4
    const val tensorManifestSha256 = "a2be311ed3b182c478a0482d78ee4ca05cc0575cdf16b16d9ee79355762c073a"
    const val rawManifestSha256 = "73c2beed647fd874be9a4187a872d33455a4a5dacf1e2b1180582a74b8a0a1ac"
    const val inputLockSha256 = "1380b926aaf7dd144bec50c7194dd6b3a31428ae1bb0da7bba121ce595f697c8"
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

    fun artifact(variant: String): ArtifactSpec = when (variant) {
        "int8" -> ArtifactSpec("int8", "r07_s1_xnnpack_int8.pte", artifactSha256, artifactBytes)
        "fp32" -> ArtifactSpec("fp32", "r07_s1_xnnpack_fp32.pte", fp32ArtifactSha256, fp32ArtifactBytes)
        else -> error("TRACKC_UNKNOWN_ARTIFACT_VARIANT:$variant")
    }

    fun verifyArtifact(file: File, spec: ArtifactSpec) {
        require(file.isFile) { "TRACKC_ARTIFACT_UNAVAILABLE:${file.absolutePath}" }
        require(file.length() == spec.bytes) { "TRACKC_ARTIFACT_BYTES_MISMATCH" }
        require(sha256(file) == spec.sha256) { "TRACKC_ARTIFACT_SHA256_MISMATCH" }
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
