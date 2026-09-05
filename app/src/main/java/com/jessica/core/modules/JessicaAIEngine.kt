package com.jessica.core.modules


class JessicaAIEngine : AIEngine {

    override suspend fun solve(
        task: String
    ): AIResult {

        if (task.isBlank()) {

            return AIResult(
                success = false,
                text = "Задача пустая"
            )

        }


        return AIResult(
            success = false,
            text =
                "AI Engine готов. " +
                "Подключение к серверу Jessica ещё не настроено."
        )

    }

}
