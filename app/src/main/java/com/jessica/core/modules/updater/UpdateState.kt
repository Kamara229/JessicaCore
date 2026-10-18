package com.jessica.core.modules.updater

sealed class UpdateState {
    object Idle : UpdateState()
    object Checking : UpdateState()
    data class UpToDate(val version: String) : UpdateState()
    data class Downloading(
        val progress: Int,
        val downloadedBytes: Long,
        val totalBytes: Long
    ) : UpdateState()
    object Verifying : UpdateState()
    data class ReadyToInstall(val version: String) : UpdateState()
    data class WaitingPermission(val version: String) : UpdateState()
    data class Installing(val version: String) : UpdateState()
    data class Installed(val version: String) : UpdateState()
    data class Error(val message: String) : UpdateState()
}
