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
         * Сейчас тестовый режим.
         *
         * В будущем:
         *
         * Android
         *     |
         *     ↓
         * Render API
         *     |
         *     ↓
         * Проверка версии Jessica
         *
         */


    }



    fun startUpdate() {


        _state.value =
            UpdateState.Downloading(
                progress = 0
            )


        /*
         * Позже здесь будет:
         *
         * 1. Получение ссылки на обновление
         * 2. Загрузка файла
         * 3. Передача прогресса
         * 4. Проверка целостности
         */


    }



    fun updateProgress(
        progress: Int
    ) {


        _state.value =
            UpdateState.Downloading(
                progress
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



    fun fail(
        message: String
    ) {


        _state.value =
            UpdateState.Error(
                message
            )

    }



    fun reset() {


        _state.value =
            UpdateState.Idle

    }


}
