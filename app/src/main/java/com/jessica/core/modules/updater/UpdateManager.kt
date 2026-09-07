package com.jessica.core.modules.updater


import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow



class UpdateManager {


    private val _state =
        MutableStateFlow<UpdateState>(
            UpdateState.Idle
        )


    val state: StateFlow<UpdateState> =
        _state.asStateFlow()



    fun checkForUpdates() {


        _state.value =
            UpdateState.Checking


        /*
         * Пока заглушка.
         *
         * Позже здесь будет:
         *
         * Android
         *    |
         *    ↓
         * Render API
         *    |
         *    ↓
         * Проверка версии
         *
         */


    }



    fun startUpdate() {


        /*
         * Пока демонстрация процесса.
         *
         * Реальная загрузка будет позже.
         */


        _state.value =
            UpdateState.Downloading(
                progress = 0
            )


    }



    fun completeUpdate(
        version: String
    ) {


        _state.value =
            UpdateState.Completed(
                version
            )

    }



    fun error(
        message: String
    ) {


        _state.value =
            UpdateState.Error(
                message
            )

    }



}
