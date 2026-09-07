package com.jessica.core.modules.updater


import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.StateFlow



class UpdateViewModel : ViewModel() {


    private val updateManager =
        UpdateManager()



    val state: StateFlow<UpdateState> =
        updateManager.state



    fun checkUpdate() {

        updateManager
            .checkForUpdates()

    }



    fun startUpdate() {

        updateManager
            .startUpdate()

    }



    fun reset() {

        updateManager
            .reset()

    }


}
