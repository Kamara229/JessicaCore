package com.jessica.core.modules.updater

/** The independent status of the proposed local MODEL-WEIGHTS update, never APK. */
sealed class MemoryUpdateState {
    object Idle : MemoryUpdateState()
    object Checking : MemoryUpdateState()
    data class NoApprovedModel(
        val filesystemFreeBytes: Long,
        val appModelBytes: Long,
        val budgetBytes: Long
    ) : MemoryUpdateState()
    data class Error(val description: String) : MemoryUpdateState()
}
