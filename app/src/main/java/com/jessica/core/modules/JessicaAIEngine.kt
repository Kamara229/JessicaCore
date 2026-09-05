package com.jessica.core.modules

import com.jessica.core.BuildConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL


class JessicaAIEngine : AIEngine {

    companion object {

        private const val BACKEND_URL =
            "https://jessicacore.onrender.com/api/solve"

    }


    override suspend fun solve(
        task: String
    ): AIResult {

        if (task.isBlank()) {

            return AIResult(
                success = false,
                text = "Задача пустая"
            )

        }


        if (BuildConfig.JESSICA_APP_TOKEN.isBlank()) {

            return AIResult(
                success = false,
                text =
                    "Ошибка конфигурации: JESSICA_APP_TOKEN отсутствует"
            )

        }


        return withContext(
            Dispatchers.IO
        ) {

            try {

                sendRequest(
                    task
                )

            } catch (e: Exception) {

                AIResult(
                    success = false,
                    text =
                        "Ошибка подключения к AI серверу: ${
                            e.message
                                ?: "неизвестная ошибка"
                        }"
                )

            }

        }

    }


    private fun sendRequest(
        task: String
    ): AIResult {

        val url =
            URL(
                BACKEND_URL
            )


        val connection =
            url.openConnection()
                as HttpURLConnection


        try {

            connection.requestMethod =
                "POST"


            connection.connectTimeout =
                30_000


            connection.readTimeout =
                120_000


            connection.doOutput =
                true


            connection.setRequestProperty(
                "Content-Type",
                "application/json; charset=UTF-8"
            )


            connection.setRequestProperty(
                "Accept",
                "application/json"
            )


            connection.setRequestProperty(
                "X-Jessica-Token",
                BuildConfig.JESSICA_APP_TOKEN
            )


            val requestJson =
                JSONObject().apply {

                    put(
                        "task",
                        task
                    )

                }


            connection
                .outputStream
                .bufferedWriter(
                    Charsets.UTF_8
                )
                .use { writer ->

                    writer.write(
                        requestJson.toString()
                    )

                    writer.flush()

                }


            val responseCode =
                connection.responseCode


            val responseText =
                if (
                    responseCode in 200..299
                ) {

                    connection
                        .inputStream
                        .bufferedReader(
                            Charsets.UTF_8
                        )
                        .use {
                            it.readText()
                        }

                } else {

                    connection
                        .errorStream
                        ?.bufferedReader(
                            Charsets.UTF_8
                        )
                        ?.use {
                            it.readText()
                        }
                        ?: ""

                }


            if (
                responseCode !in 200..299
            ) {

                val serverMessage =
                    try {

                        JSONObject(
                            responseText
                        ).optString(
                            "text",
                            ""
                        )

                    } catch (e: Exception) {

                        ""

                    }


                return AIResult(
                    success = false,
                    text =
                        when {

                            responseCode == 401 ->
                                "Ошибка авторизации Jessica"

                            serverMessage.isNotBlank() ->
                                serverMessage

                            else ->
                                "Ошибка сервера: HTTP $responseCode"

                        }
                )

            }


            val json =
                JSONObject(
                    responseText
                )


            val success =
                json.optBoolean(
                    "success",
                    false
                )


            val text =
                json.optString(
                    "text",
                    ""
                )


            if (!success) {

                return AIResult(
                    success = false,
                    text =
                        if (
                            text.isNotBlank()
                        ) {

                            text

                        } else {

                            "AI сервер вернул ошибку"

                        }
                )

            }


            if (text.isBlank()) {

                return AIResult(
                    success = false,
                    text =
                        "AI сервер вернул пустой ответ"
                )

            }


            return AIResult(
                success = true,
                text = text
            )

        } finally {

            connection.disconnect()

        }

    }

}
