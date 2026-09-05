package com.jessica.core.modules

import android.content.Context


data class TaskExecutionResult(
    val success: Boolean,
    val result: String
)


class TaskExecutor(
    context: Context
) {

    private val eventStorage =
        EventStorage(context)

    private val aiEngine: AIEngine =
        JessicaAIEngine()


    suspend fun execute(
        task: JessicaTask
    ): TaskExecutionResult {

        if (task.text.isBlank()) {

            return TaskExecutionResult(
                success = false,
                result = "Текст задачи пуст"
            )

        }


        eventStorage.saveEvent(
            type = "task_executor",
            message = "Начато выполнение задачи ${task.id}"
        )


        val aiResult =
            aiEngine.solve(
                task.text
            )


        if (!aiResult.success) {

            eventStorage.saveEvent(
                type = "error",
                message =
                    "Ошибка AI Engine при выполнении задачи ${task.id}"
            )


            return TaskExecutionResult(
                success = false,
                result = aiResult.text
            )

        }


        eventStorage.saveEvent(
            type = "task_executor",
            message = "Завершено выполнение задачи ${task.id}"
        )


        return TaskExecutionResult(
            success = true,
            result = aiResult.text
        )

    }

}
