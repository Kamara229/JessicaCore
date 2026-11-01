package com.jessica.core.modules.updater

import org.junit.Assert.*
import org.junit.Test

class ModelReleaseGateTest {
    private val manifest = "a".repeat(64)
    private val weightHash = "b".repeat(64)
    private val twoGiB = 2L * 1024L * 1024L * 1024L

    private fun artifact(
        fileBytes: Long = twoGiB,
        compressionLoss: Double = 0.01,
        speedRatio: Double = 1.01
    ) = ModelReleaseGate.Artifact(
        modelId = "approved-model", version = "v1",
        format = "CLB8-CPU-v1", fileBytes = fileBytes,
        weightsSha256 = weightHash, signedManifestSha256 = manifest,
        qualityBaselineNll = 2.0, qualityCompressedNll = 2.0 + compressionLoss,
        measuredRuntimeRatio = speedRatio, modelRuntimeAbi = 1
    )

    private fun device(
        diskFree: Long = 90_000_000_000L,
        existing: Long = 0L
    ) = ModelReleaseGate.Device(
        availableFilesystemBytes = diskFree,
        existingModelStorageBytes = existing,
        supportedRuntimeAbi = 1,
        supportedFormats = setOf("CLB8-CPU-v1")
    )

    @Test fun denyUnsignedEvenWhenEveryMetricIsGood() {
        assertTrue(ModelReleaseGate.check(artifact(), device()) is ModelReleaseGate.Decision.Rejected)
    }

    @Test fun allowOnlyPreapprovedCompatibleArtifactWithinActualDiskAndQuota() {
        val r = ModelReleaseGate.check(artifact(), device(), setOf(manifest))
        assertTrue(r is ModelReleaseGate.Decision.Eligible)
    }

    @Test fun rejectQualityRegression() {
        assertTrue(ModelReleaseGate.check(artifact(compressionLoss=0.1),
            device(),setOf(manifest)) is ModelReleaseGate.Decision.Rejected)
    }

    @Test fun rejectUnsupportedReaderOrSlowInference() {
        assertTrue(ModelReleaseGate.check(artifact(speedRatio=1.10),
            device(),setOf(manifest)) is ModelReleaseGate.Decision.Rejected)
        val broken=device().copy(supportedRuntimeAbi=2)
        assertTrue(ModelReleaseGate.check(artifact(),
            broken,setOf(manifest)) is ModelReleaseGate.Decision.Rejected)
    }

    @Test fun rejectEvenWithReserved80GBIfFilesystemActuallyInsufficient() {
        assertTrue(ModelReleaseGate.check(artifact(),
            device(diskFree=twoGiB + 2L*1024L*1024L*1024L),
            setOf(manifest)) is ModelReleaseGate.Decision.Rejected)
    }

    @Test fun rejectStagingIfTwoModelsExceedTotalBudget() {
        assertTrue(ModelReleaseGate.check(
            artifact(fileBytes=5_000_000_000L),
            device(existing=77_000_000_000L),
            setOf(manifest)) is ModelReleaseGate.Decision.Rejected)
    }

    @Test fun rejectMalformedHashPathAndNegativeOrOverflowLength() {
        val x=artifact().copy(modelId="../escape")
        assertTrue(ModelReleaseGate.check(x,device(),setOf(manifest)) is ModelReleaseGate.Decision.Rejected)
        val y=artifact().copy(weightsSha256="1" + weightHash)
        assertTrue(ModelReleaseGate.check(y,device(),setOf(manifest)) is ModelReleaseGate.Decision.Rejected)
        val z=artifact(fileBytes=Long.MAX_VALUE)
        assertTrue(ModelReleaseGate.check(z,device(),setOf(manifest)) is ModelReleaseGate.Decision.Rejected)
    }
}
