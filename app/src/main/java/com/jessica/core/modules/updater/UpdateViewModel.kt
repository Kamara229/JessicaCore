package com.jessica.core.modules.updater

import android.content.Context
import android.content.Intent
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.StateFlow

class UpdateViewModel(context: Context) : ViewModel() {
    private val updateManager = UpdateManager(context.applicationContext)
    private val memoryUpdateManager = MemoryUpdateManager(context.applicationContext)
    val state: StateFlow<UpdateState> = updateManager.state
    val memoryState: StateFlow<MemoryUpdateState> = memoryUpdateManager.state

    fun checkUpdate() = updateManager.checkForUpdates(viewModelScope)
    fun checkMemoryUpdate() = memoryUpdateManager.checkForMemoryUpdates(viewModelScope)
    fun prepareInstallation(): Intent? = updateManager.prepareInstallation()
    fun waitingPermission() = updateManager.waitingPermission()
    fun installing() = updateManager.installing()
    fun permissionDenied() = updateManager.permissionDenied()
    fun installationReturned() = updateManager.installationReturned()
    fun reset() = updateManager.reset()
}
