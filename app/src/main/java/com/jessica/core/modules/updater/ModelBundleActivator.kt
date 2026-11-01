package com.jessica.core.modules.updater

import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest

/**
 * Research-only, offline finalization of previously downloaded model shards.
 * Input: a manifest VERIFIED by ModelReleaseGate's publisher allowlist and
 * shards downloaded separately to app-private staging on the same volume.
 *
 * Never unpacks another full 80GB copy. No update-button wiring, no network,
 * no model runtime; work remains gated until a verified compatible release.
 */
internal object ModelBundleActivator {
    data class Shard(val fileName: String, val bytes: Long, val sha256: String)
    private val idPattern = Regex("[A-Za-z0-9._-]{1,128}")
    private val hashPattern = Regex("[0-9a-f]{64}")

    private fun sha256(file: File): String {
        val dig = MessageDigest.getInstance("SHA-256")
        FileInputStream(file).use { input ->
            val buffer=ByteArray(1024 * 1024)
            while (true) {
                val count=input.read(buffer)
                if(count<0)break
                dig.update(buffer,0,count)
            }
        }
        return dig.digest().joinToString(""){"%02x".format(it)}
    }

    /**
     * Caller must check ModelReleaseGate.Decision.Eligible immediately before
     * entering this method; this is *not* a replacement for authenticated
     * remote manifest signature verification.
     */
    fun activateVerifiedStagedBundle(
        modelsRoot: File,
        stagingDir: File,
        modelId: String,
        version: String,
        shards: List<Shard>,
        approval: ModelReleaseGate.Decision.Eligible
    ): File {
        if (!idPattern.matches(modelId) || !idPattern.matches(version) ||
            shards.isEmpty() || shards.any {
                !idPattern.matches(it.fileName) ||
                !hashPattern.matches(it.sha256) || it.bytes<=0
            } || shards.map { it.fileName }.distinct().size!=shards.size) {
            throw IOException("Invalid trusted model identity or shard manifest")
        }
        if (!stagingDir.isDirectory || !modelsRoot.isDirectory ||
            stagingDir.canonicalFile.parentFile != modelsRoot.canonicalFile) {
            throw IOException("Staging must be directly under app-owned model root")
        }
        if (Files.isSymbolicLink(stagingDir.toPath())) {
            throw IOException("Symlink staging directory is not allowed")
        }
        val total=shards.fold(0L){sum,shard->
            if(sum>Long.MAX_VALUE-shard.bytes)throw IOException("Size overflow")
            sum+shard.bytes
        }
        if(total>approval.totalAllocatedBytesAfterStaging) {
            throw IOException("Staging size exceeds approved amount")
        }
        for (shard in shards) {
            val file=File(stagingDir,shard.fileName)
            if(Files.isSymbolicLink(file.toPath()) || !file.isFile ||
                file.canonicalFile.parentFile != stagingDir.canonicalFile) {
                throw IOException("Missing, unsafe or nonregular shard")
            }
            if(file.length()!=shard.bytes || sha256(file)!=shard.sha256) {
                throw IOException("Shard integrity check failed: ${shard.fileName}")
            }
        }
        val versionDir=File(modelsRoot,"${modelId}-${version}")
        if(versionDir.exists())throw IOException("Version already present")
        try {
            Files.move(stagingDir.toPath(),versionDir.toPath(),StandardCopyOption.ATOMIC_MOVE)
        } catch (e: AtomicMoveNotSupportedException) {
            throw IOException("Atomic model-directory activation not supported", e)
        }
        val pointerTemp=File(modelsRoot,"active_model.tmp")
        val pointer=File(modelsRoot,"active_model")
        FileOutputStream(pointerTemp).use { out ->
            out.write(versionDir.name.toByteArray(Charsets.UTF_8))
            out.fd.sync()
        }
        try {
            Files.move(pointerTemp.toPath(),pointer.toPath(),
                StandardCopyOption.REPLACE_EXISTING,StandardCopyOption.ATOMIC_MOVE)
        } catch (e: AtomicMoveNotSupportedException) {
            throw IOException("Atomic model pointer update not supported",e)
        }
        return versionDir
    }
}
