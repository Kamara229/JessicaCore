package com.jessica.core.modules.updater

import org.junit.Assert.*
import org.junit.Test

class UpdateReleasePolicyTest {
    @Test fun stableChannelOnlyAcceptsOfficialNonBetaTags() {
        assertEquals(2518L, UpdateReleasePolicy.versionCode("v0.1.2518", UpdateChannel.STABLE))
        assertNull(UpdateReleasePolicy.versionCode("v0.1.2520-beta", UpdateChannel.STABLE))
        assertNull(UpdateReleasePolicy.versionCode("v0.1.2520-CALIBRI-beta", UpdateChannel.STABLE))
        assertNull(UpdateReleasePolicy.versionCode("v0.1.x", UpdateChannel.STABLE))
    }

    @Test fun betaMustBeExplicitAndVersioned() {
        assertEquals(2520L, UpdateReleasePolicy.versionCode("v0.1.2520-beta", UpdateChannel.BETA))
        assertNull(UpdateReleasePolicy.versionCode("v0.1.2518", UpdateChannel.BETA))
        assertNull(UpdateReleasePolicy.versionCode("v0.1.2520-beta-extra", UpdateChannel.BETA))
        assertNull(UpdateReleasePolicy.versionCode("v0.1.0-beta", UpdateChannel.BETA))
        assertNull(UpdateReleasePolicy.versionCode("v0.1.9999999999999999-beta", UpdateChannel.BETA))
    }

    @Test fun channelsUseDistinctReleaseAssetNames() {
        assertEquals("JessicaCore.apk", UpdateReleasePolicy.apkAssetName(UpdateChannel.STABLE))
        assertEquals("JessicaCore.apk.sha256", UpdateReleasePolicy.checksumName(UpdateChannel.STABLE))
        assertEquals("JessicaCore-CALIBRI-beta.apk", UpdateReleasePolicy.apkAssetName(UpdateChannel.BETA))
        assertEquals("JessicaCore-CALIBRI-beta.apk.sha256", UpdateReleasePolicy.checksumName(UpdateChannel.BETA))
    }
}
