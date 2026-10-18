package com.jessica.core.modules.updater

import android.content.Context
import android.content.Intent
import androidx.core.content.FileProvider
import java.io.File

internal object UpdateInstaller {
    fun installationIntent(context: Context, apk: File): Intent {
        val uri = FileProvider.getUriForFile(
            context,
            "${context.packageName}.updates",
            apk
        )
        return Intent(Intent.ACTION_INSTALL_PACKAGE).apply {
            data = uri
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            putExtra(Intent.EXTRA_RETURN_RESULT, true)
        }
    }
}
