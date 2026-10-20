package com.jessica.core.modules.chat

import android.content.Context
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters

/** Purge expired local chats and their attachments after the grace period. */
class ChatPurgeWorker(
    context: Context,
    params: WorkerParameters
) : CoroutineWorker(context, params) {
    override suspend fun doWork(): Result =
        try {
            ChatThreadStore(applicationContext).purgeExpired()
            Result.success()
        } catch (_: Exception) {
            Result.retry()
        }
}
