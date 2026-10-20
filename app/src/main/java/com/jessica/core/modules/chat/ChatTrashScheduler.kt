package com.jessica.core.modules.chat

import android.content.Context
import androidx.work.ExistingWorkPolicy
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import java.util.concurrent.TimeUnit

/** Unique device-local cleanup job: not allowed to run before 60 minutes. */
object ChatTrashScheduler {
    private fun name(id: String) = "jessica-trash-" + id

    fun schedule(context: Context, id: String) {
        val request = OneTimeWorkRequestBuilder<ChatPurgeWorker>()
            .setInitialDelay(ChatTrashPolicy.RETENTION_MS, TimeUnit.MILLISECONDS)
            .build()
        WorkManager.getInstance(context).enqueueUniqueWork(
            name(id), ExistingWorkPolicy.REPLACE, request
        )
    }

    fun cancel(context: Context, id: String) {
        WorkManager.getInstance(context).cancelUniqueWork(name(id))
    }
}
