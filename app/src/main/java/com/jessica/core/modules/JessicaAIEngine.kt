package com.jessica.core.modules

import com.jessica.core.BuildConfig
import com.jessica.core.modules.chat.ChatMessage
import com.jessica.core.modules.chat.DocumentContext
import com.jessica.core.modules.chat.ChatContextWindow
import org.json.JSONArray
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.SocketTimeoutException
import java.net.URL


class JessicaAIEngine(
    private val languageProvider: () -> String = { "ru" }
) : AIEngine {

    private fun tr(ru: String, en: String): String =
        if (languageProvider() == "en") en else ru

    companion object {

        private const val BACKEND_URL =
            "https://jessicacore.onrender.com/api/solve"


        /*
         * Время на установку соединения.
         */
        private const val CONNECT_TIMEOUT_MS =
            45_000


        /*
         * Максимальное ожидание выполнения задачи.
         *
         * Сложная задача Jessica может пройти:
         *
         * Planner
         * → Tools
         * → Composer
         * → Validator
         * → Replanner
         * → повторное выполнение
         *
         * Поэтому 120 секунд уже недостаточно.
         */
        private const val READ_TIMEOUT_MS =
            360_000

    }


    override suspend fun solve(task: String): AIResult = solve(task, emptyList())

    suspend fun solve(
        task: String,
        history: List<ChatMessage>,
        documents: List<DocumentContext> = emptyList()
    ): AIResult {

        if (task.isBlank()) {

            return AIResult(
                success = false,
                text = tr("Задача пустая", "The task is empty")
            )

        }


        if (BuildConfig.JESSICA_APP_TOKEN.isBlank()) {

            return AIResult(
                success = false,
                text =
                    tr("Ошибка конфигурации: JESSICA_APP_TOKEN отсутствует", "Configuration error: JESSICA_APP_TOKEN is missing")
            )

        }


        return withContext(
            Dispatchers.IO
        ) {

            try {

                sendRequest(
                    task,
                    history,
                    documents
                )

            } catch (e: SocketTimeoutException) {

                AIResult(
                    success = false,
                    text =
                        tr("Jessica не успела завершить задачу за отведённое время", "Jessica could not complete the task before the timeout")
                )

            } catch (e: Exception) {

                AIResult(
                    success = false,
                    text =
                        tr("Ошибка подключения к AI серверу: ", "Could not connect to the AI server: ") + "${
                            e.message
                                ?: tr("неизвестная ошибка", "unknown error")
                        }"
                )

            }

        }

    }


    private fun sendRequest(
        task: String,
        history: List<ChatMessage>,
        documents: List<DocumentContext>
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
                CONNECT_TIMEOUT_MS


            connection.readTimeout =
                READ_TIMEOUT_MS


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
                    put("outputLanguage", languageProvider().takeIf { it == "en" } ?: "ru")

                    val conversation = JSONArray()
                    ChatContextWindow.select(history).forEach { item ->
                        conversation.put(JSONObject().apply {
                            put("role", if (item.isUser) "user" else "assistant")
                            put("content", item.text)
                        })
                    }
                    put("conversation", conversation)

                    val attachments = JSONArray()
                    documents.forEach { document ->
                        attachments.put(JSONObject().apply {
                            put("name", document.name)
                            put("text", document.text)
                            put("truncated", document.truncated)
                        })
                    }
                    put("attachments", attachments)

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
                                tr("Ошибка авторизации Jessica", "Jessica authorization failed")

                            serverMessage.isNotBlank() ->
                                serverMessage

                            else ->
                                tr("Ошибка сервера: HTTP ", "Server error: HTTP ") + responseCode

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

                            tr("AI сервер вернул ошибку", "AI server returned an error")

                        }
                )

            }


            if (text.isBlank()) {

                return AIResult(
                    success = false,
                    text =
                        tr("AI сервер вернул пустой ответ", "AI server returned an empty answer")
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
