package com.jessica.core.modules.codex.chat

import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities

/**
 * Read-only local connectivity signal. Does not ping third-party hosts, change
 * Wi-Fi/VPN settings, retry an uncertain send or consume user message content.
 */
internal object CodexNetworkObserver {
    fun offline(context: Context): Boolean {
        val connectivity=context.getSystemService(ConnectivityManager::class.java)
            ?: return false
        val network=connectivity.activeNetwork ?: return true
        val capabilities=connectivity.getNetworkCapabilities(network) ?: return false
        return !capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
    }
}
