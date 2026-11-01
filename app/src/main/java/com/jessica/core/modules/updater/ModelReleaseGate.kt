package com.jessica.core.modules.updater

/**
 * Quality- and space-gated, OFFLINE model asset admission policy for future
 * update-button integration. This deliberately neither downloads nor installs
 * model files. APK updates must keep using their separate repository/installer.
 *
 * The 80 GB is a USER RESERVATION / hard budget, NOT evidence of free disk.
 * The default publisherDigestAllowlist is EMPTY: no weight download is
 * authorized until a separately reviewed signed model release is published.
 */
internal object ModelReleaseGate {
    const val WEIGHTS_BUDGET_BYTES: Long = 80_000_000_000L // decimal GB
    const val OS_HEADROOM_BYTES: Long = 4L * 1024L * 1024L * 1024L

    data class Artifact(
        val modelId: String,
        val version: String,
        val format: String,
        val fileBytes: Long,
        val weightsSha256: String,
        val signedManifestSha256: String,
        val qualityBaselineNll: Double,
        val qualityCompressedNll: Double,
        val measuredRuntimeRatio: Double,
        val modelRuntimeAbi: Int
    )

    data class Device(
        val availableFilesystemBytes: Long,
        val existingModelStorageBytes: Long,
        val supportedRuntimeAbi: Int,
        val supportedFormats: Set<String>
    )

    sealed class Decision {
        data class Eligible(val totalAllocatedBytesAfterStaging: Long): Decision()
        data class Rejected(val reason: String): Decision()
    }

    private val HASH = Regex("[0-9a-f]{64}")
    private val MODEL = Regex("[a-zA-Z0-9._-]{1,128}")

    fun check(
        a: Artifact,
        d: Device,
        trustedManifestDigests: Set<String> = emptySet()
    ): Decision {
        // Authentication must be enforced by an actual trusted verifier/pinned
        // release digest. Self-declared quality or source URL is NOT enough.
        if (!HASH.matches(a.signedManifestSha256) ||
            a.signedManifestSha256 !in trustedManifestDigests) {
            return Decision.Rejected("Model manifest is not independently approved")
        }
        if (!MODEL.matches(a.modelId) || !MODEL.matches(a.version) ||
            !HASH.matches(a.weightsSha256)) {
            return Decision.Rejected("Invalid model identity or expected SHA-256")
        }
        if (a.format !in d.supportedFormats || a.modelRuntimeAbi != d.supportedRuntimeAbi) {
            return Decision.Rejected("No compatible verified on-device model reader")
        }
        if (!a.qualityBaselineNll.isFinite() || !a.qualityCompressedNll.isFinite() ||
            a.qualityBaselineNll <= 0 || a.qualityCompressedNll <= 0 ||
            !a.measuredRuntimeRatio.isFinite() || a.measuredRuntimeRatio <= 0) {
            return Decision.Rejected("Incomplete independent evaluation")
        }
        // Exp(delta NLL) is a language-model perplexity ratio. The actual
        // evaluator must use identical held-out data/context/weights.
        if (a.qualityCompressedNll - a.qualityBaselineNll > kotlin.math.ln(1.05)) {
            return Decision.Rejected("Loss exceeds approved 5% perplexity budget")
        }
        if (a.measuredRuntimeRatio > 1.05) {
            return Decision.Rejected("Inference runtime exceeds 5% budget")
        }
        if (a.fileBytes <= 0 || d.existingModelStorageBytes < 0 ||
            a.fileBytes > WEIGHTS_BUDGET_BYTES ||
            d.existingModelStorageBytes > WEIGHTS_BUDGET_BYTES - a.fileBytes) {
            return Decision.Rejected("Staged model and existing models exceed 80 GB budget")
        }
        if (d.availableFilesystemBytes < OS_HEADROOM_BYTES ||
            a.fileBytes > d.availableFilesystemBytes - OS_HEADROOM_BYTES) {
            return Decision.Rejected("Insufficient actual disk space including OS reserve")
        }
        return Decision.Eligible(d.existingModelStorageBytes + a.fileBytes)
    }
}
