package com.jessica.core.modules.updater

/** Stable releases and opt-in beta releases are intentionally separate. */
enum class UpdateChannel { STABLE, BETA }

internal object UpdateReleasePolicy {
    private val stableTag = Regex("^v0\\.1\\.([0-9]+)$")
    private val betaTag = Regex("^v0\\.1\\.([0-9]+)-beta$")

    fun versionCode(tag: String, channel: UpdateChannel): Long? =
        (if (channel == UpdateChannel.STABLE) stableTag else betaTag)
            .matchEntire(tag)?.groupValues?.get(1)?.toLongOrNull()
                ?.takeIf { it in 1..2_100_000_000 }

    fun apkAssetName(channel: UpdateChannel): String =
        if (channel == UpdateChannel.BETA) "JessicaCore-CALIBRI-beta.apk" else "JessicaCore.apk"

    fun checksumName(channel: UpdateChannel): String = apkAssetName(channel) + ".sha256"
}
