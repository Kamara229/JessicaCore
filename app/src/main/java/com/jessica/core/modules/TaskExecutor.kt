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


    fun execute(
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


        val result =
            solveLocally(
                task.text
            )


        eventStorage.saveEvent(
            type = "task_executor",
            message = "Завершено выполнение задачи ${task.id}"
        )


        return TaskExecutionResult(
            success = true,
            result = result
        )

    }


    private fun solveLocally(
        taskText: String
    ): String {

        return buildString {

            appendLine(
                "Task Executor получил задачу:"
            )

            appendLine()

            appendLine(
                taskText
            )

            appendLine()

            append(
                "Для полноценного решения задачи требуется подключение AI Engine."
            )

        }

    }

}
