package com.jessica.core.modules.updater

import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.StateFlow


class UpdateViewModel : ViewModel() {


    private val manager =
        UpdateManager()


    val state: StateFlow<UpdateState> =
        manager.state



    fun checkUpdate() {

        manager.checkForUpdates()

    }


    fun startUpdate() {

        manager.startUpdate()

    }

}
