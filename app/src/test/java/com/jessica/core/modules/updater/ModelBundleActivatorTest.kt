package com.jessica.core.modules.updater

import java.io.File
import java.io.IOException
import java.security.MessageDigest
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class ModelBundleActivatorTest {
    @get:Rule val temp = TemporaryFolder()

    private fun digest(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(bytes).joinToString(""){"%02x".format(it)}

    private fun eligible(size:Long):ModelReleaseGate.Decision.Eligible {
        val hash="e".repeat(64)
        val a=ModelReleaseGate.Artifact(
            "model123","v1","CLB8-CPU-v1",size,digest("weights".toByteArray()),
            hash,2.0,2.01,1.0,1)
        val d=ModelReleaseGate.Device(
            90_000_000_000L,0,1,setOf("CLB8-CPU-v1"))
        val result=ModelReleaseGate.check(a,d,setOf(hash))
        assertTrue(result is ModelReleaseGate.Decision.Eligible)
        return result as ModelReleaseGate.Decision.Eligible
    }

    @Test fun validVerifiedStagedBundleAtomicallyChangesActivePointer() {
        val root=temp.newFolder("models")
        val staging=File(root,"staging").apply { mkdir() }
        val data=ByteArray(123) { (it and 0xff).toByte() }
        File(staging,"weights0001").writeBytes(data)
        val shard=ModelBundleActivator.Shard("weights0001",data.size.toLong(),digest(data))
        val installed=ModelBundleActivator.activateVerifiedStagedBundle(
            root,staging,"model123","v1",listOf(shard),eligible(data.size.toLong()))
        assertTrue(File(installed,"weights0001").isFile)
        assertFalse(staging.exists())
        assertEquals("model123-v1",File(root,"active_model").readText())
    }

    @Test fun corruptedShardMustNeverChangeOldActivePointer() {
        val root=temp.newFolder("models")
        File(root,"active_model").writeText("old-model-v1")
        val staging=File(root,"staging").apply { mkdir() }
        val data=ByteArray(64){it.toByte()}
        File(staging,"weights0001").writeBytes(data)
        val bad=ModelBundleActivator.Shard("weights0001",64,"0".repeat(64))
        try {
            ModelBundleActivator.activateVerifiedStagedBundle(
                root,staging,"model123","v1",listOf(bad),eligible(64))
            fail("Corrupted shard should not activate")
        } catch (_: IOException) { }
        assertEquals("old-model-v1",File(root,"active_model").readText())
        assertTrue(staging.isDirectory)
    }

    @Test fun disallowTraversalAndMissingShards() {
        val root=temp.newFolder("models")
        val staging=File(root,"staging").apply{mkdir()}
        val wrong=ModelBundleActivator.Shard("../x",4,"a".repeat(64))
        try {
            ModelBundleActivator.activateVerifiedStagedBundle(
                root,staging,"model123","v1",listOf(wrong),eligible(4))
            fail("Path traversal must reject")
        } catch (_:IOException) { }
        assertFalse(File(root,"active_model").exists())
    }
}
