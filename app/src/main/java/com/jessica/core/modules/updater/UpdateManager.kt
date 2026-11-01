package com.jessica.core.modules.updater

import android.content.Context
import android.content.Intent
import com.jessica.core.BuildConfig
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class UpdateManager(private val context: Context) {
    private val repository = UpdateRepository(context)
    private val preferences =
        context.getSharedPreferences("jessica_update", Context.MODE_PRIVATE)
    private val _state = MutableStateFlow<UpdateState>(initialState())
    val state: StateFlow<UpdateState> = _state.asStateFlow()

    private var job: Job? = null
    private var downloadedApk: File? = null
    private var target: UpdateRelease? = null
    private var requestedChannel: UpdateChannel? = null

    private fun initialState(): UpdateState {
        val pending = preferences.getLong("pending_version_code", -1)
        return if (pending > 0 && BuildConfig.VERSION_CODE.toLong() >= pending) {
            preferences.edit().remove("pending_version_code").apply()
            UpdateState.Installed(BuildConfig.VERSION_NAME)
        } else {
            UpdateState.Idle
        }
    }

    fun checkForUpdates(
        scope: CoroutineScope,
        channel: UpdateChannel = UpdateChannel.STABLE
    ) {
        if (job?.isActive == true) return
        // Do not silently install a cached APK from the OTHER channel.
        if (requestedChannel != channel) {
            downloadedApk = null
            target = null
            requestedChannel = channel
        }
        if (downloadedApk?.exists() == true && target != null) {
            _state.value = UpdateState.ReadyToInstall(target!!.versionName)
            return
        }
        job = scope.launch {
            _state.value = UpdateState.Checking
            try {
                val latest = withContext(Dispatchers.IO) {
                    repository.latestRelease(channel)
                }
                if (latest.versionCode <= BuildConfig.VERSION_CODE.toLong()) {
                    _state.value = UpdateState.UpToDate(BuildConfig.VERSION_NAME)
                    return@launch
                }
                target = latest
                _state.value = UpdateState.Downloading(0, 0, latest.size)
                val apk = withContext(Dispatchers.IO) {
                    repository.download(
                        latest,
                        onProgress = { percent, bytes, total ->
                            _state.value = UpdateState.Downloading(percent, bytes, total)
                        },
                        onVerifying = {
                            _state.value = UpdateState.Verifying
                        }
                    )
                }
                downloadedApk = apk
                _state.value = UpdateState.ReadyToInstall(latest.versionName)
            } catch (error: Exception) {
                _state.value = UpdateState.Error(
                    error.message ?: "Не удалось скачать обновление"
                )
            }
        }
    }

    /** Preserve selected Beta vs Stable channel when Android grants APK permission. */
    fun resumeChosenUpdate(scope: CoroutineScope) =
        checkForUpdates(scope, requestedChannel ?: UpdateChannel.STABLE)

    fun prepareInstallation(): Intent? {
        val apk = downloadedApk
        val release = target
        if (apk == null || !apk.isFile || release == null) {
            _state.value = UpdateState.Error("Файл обновления недоступен")
            return null
        }
        return try {
            UpdateInstaller.installationIntent(context, apk)
        } catch (error: Exception) {
            _state.value = UpdateState.Error(
                error.message ?: "Не удалось запустить установку"
            )
            null
        }
    }

    fun waitingPermission() {
        target?.let { _state.value = UpdateState.WaitingPermission(it.versionName) }
    }

    fun installing() {
        target?.let {
            preferences.edit().putLong("pending_version_code", it.versionCode).apply()
            _state.value = UpdateState.Installing(it.versionName)
        }
    }

    fun permissionDenied() {
        _state.value = UpdateState.Error(
            "Разрешите установку обновлений для Jessica Core в настройках Android"
        )
    }

    fun installationReturned() {
        val release = target
        if (release != null && BuildConfig.VERSION_CODE.toLong() >= release.versionCode) {
            preferences.edit().remove("pending_version_code").apply()
            _state.value = UpdateState.Installed(BuildConfig.VERSION_NAME)
        } else {
            _state.value = UpdateState.Error(
                "Установка не завершена или отменена. Нажмите «Повторить»."
            )
        }
    }

    fun reset() {
        if (job?.isActive == true) return
        _state.value = UpdateState.Idle
    }
}
