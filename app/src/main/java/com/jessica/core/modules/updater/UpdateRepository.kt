package com.jessica.core.modules.updater

import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.os.Environment
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

internal data class UpdateRelease(
    val versionCode: Long,
    val versionName: String,
    val downloadUrl: String,
    val sha256: String,
    val size: Long
)

internal class UpdateRepository(private val context: Context) {
    private val releaseApi =
        "https://api.github.com/repos/Kamara229/JessicaCore/releases/latest"

    fun latestRelease(): UpdateRelease {
        val release = JSONObject(fetchText(releaseApi))
        val assets = release.getJSONArray("assets")
        var apk: JSONObject? = null
        var checksumUrl: String? = null
        for (index in 0 until assets.length()) {
            val entry = assets.getJSONObject(index)
            when (entry.optString("name")) {
                "JessicaCore.apk" -> apk = entry
                "JessicaCore.apk.sha256" ->
                    checksumUrl = entry.optString("browser_download_url")
            }
        }
        val asset = apk ?: throw IOException("В последнем релизе нет JessicaCore.apk")
        val digest = asset.optString("digest")
        val expectedHash = if (digest.startsWith("sha256:")) {
            digest.removePrefix("sha256:")
        } else {
            val url = checksumUrl
                ?: throw IOException("У релиза отсутствует контрольная сумма SHA-256")
            fetchText(url).trim().split(Regex("\\s+")).first()
        }
        if (!expectedHash.matches(Regex("[0-9a-fA-F]{64}"))) {
            throw IOException("Неверная контрольная сумма APK")
        }
        val name = release.optString("tag_name").removePrefix("v")
        // The signed APK is built with GitHub Actions run_number as versionCode.
        val versionCode = name.substringAfterLast('.').toLongOrNull()
            ?: throw IOException("Неверный номер версии релиза")
        return UpdateRelease(
            versionCode = versionCode,
            versionName = name,
            downloadUrl = asset.getString("browser_download_url"),
            sha256 = expectedHash.lowercase(),
            size = asset.optLong("size", -1L)
        )
    }

    fun download(
        release: UpdateRelease,
        onProgress: (Int, Long, Long) -> Unit
    ): File {
        val directory = File(context.cacheDir, "jessica_updates")
        if (!directory.exists() && !directory.mkdirs()) {
            throw IOException("Не удалось создать папку загрузки")
        }
        val partial = File(directory, "JessicaCore.apk.part")
        val ready = File(directory, "JessicaCore.apk")
        partial.delete()
        try {
            val connection = openConnection(release.downloadUrl)
            try {
                val total = if (release.size > 0) release.size
                    else connection.contentLengthLong
                if (total > 250L * 1024 * 1024) {
                    throw IOException("Размер обновления превышает допустимый")
                }
                val sha = MessageDigest.getInstance("SHA-256")
                var loaded = 0L
                connection.inputStream.use { input ->
                    FileOutputStream(partial).use { output ->
                        val buffer = ByteArray(32 * 1024)
                        while (true) {
                            val length = input.read(buffer)
                            if (length < 0) break
                            loaded += length
                            if (loaded > 250L * 1024 * 1024) {
                                throw IOException("Обновление слишком большое")
                            }
                            sha.update(buffer, 0, length)
                            output.write(buffer, 0, length)
                            val percent =
                                if (total > 0) (loaded * 100 / total).toInt().coerceIn(0, 100)
                                else 0
                            onProgress(percent, loaded, total)
                        }
                        output.fd.sync()
                    }
                }
                if (total > 0 && total != loaded) {
                    throw IOException("Загружен неполный APK")
                }
                val actualHash = sha.digest().joinToString("") { "%02x".format(it) }
                if (!actualHash.equals(release.sha256, ignoreCase = true)) {
                    throw IOException("Контрольная сумма APK не совпадает")
                }
                if (!partial.renameTo(ready)) {
                    partial.copyTo(ready, overwrite = true)
                    partial.delete()
                }
                verifyPackage(ready, release)
                return ready
            } finally {
                connection.disconnect()
            }
        } catch (error: Exception) {
            partial.delete()
            ready.delete()
            throw error
        }
    }

    private fun verifyPackage(apk: File, release: UpdateRelease) {
        @Suppress("DEPRECATION")
        val info = context.packageManager.getPackageArchiveInfo(
            apk.absolutePath, 0
        ) ?: throw IOException("Загруженный файл не является APK")
        if (info.packageName != context.packageName) {
            throw IOException("У обновления неверный идентификатор приложения")
        }
        if (info.longVersionCode != release.versionCode) {
            throw IOException("Версия APK не совпадает с версией релиза")
        }
    }

    private fun fetchText(url: String): String {
        val connection = openConnection(url)
        return try {
            connection.inputStream.bufferedReader(Charsets.UTF_8).use {
                it.readText()
            }
        } finally {
            connection.disconnect()
        }
    }

    private fun openConnection(url: String): HttpURLConnection {
        val parsed = URL(url)
        if (parsed.protocol != "https") {
            throw IOException("Обновление разрешено только по HTTPS")
        }
        val connection = parsed.openConnection() as HttpURLConnection
        connection.connectTimeout = 15000
        connection.readTimeout = 60000
        connection.instanceFollowRedirects = true
        connection.setRequestProperty("User-Agent", "JessicaCore-Android-Updater")
        connection.connect()
        if (connection.responseCode !in 200..299) {
            val status = connection.responseCode
            connection.disconnect()
            if (status == 404) {
                throw IOException("Пока нет опубликованных версий Jessica Core")
            }
            throw IOException("Сервер обновлений ответил: HTTP $status")
        }
        return connection
    }
}
