package com.jessica.core.modules

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject


data class JessicaTask(
    val id: Long,
    val text: String,
    val status: String,
    val result: String,
    val createdAt: Long
)


class TaskStorage(
    private val context: Context
) {

    private val fileName =
        "jessica_tasks.json"


    fun createTask(
        text: String
    ): JessicaTask {

        val task =
            JessicaTask(
                id = System.currentTimeMillis(),
                text = text,
                status = "NEW",
                result = "",
                createdAt = System.currentTimeMillis()
            )


        val tasks =
            loadTasks()
                .toMutableList()


        tasks.add(
            task
        )


        saveTasks(
            tasks
        )


        return task

    }


    fun loadTasks(): List<JessicaTask> {

        return try {

            val text =
                context
                    .openFileInput(fileName)
                    .bufferedReader()
                    .use {
                        it.readText()
                    }


            val array =
                JSONArray(text)


            val result =
                mutableListOf<JessicaTask>()


            for (
                i in 0 until array.length()
            ) {

                val obj =
                    array.getJSONObject(i)


                result.add(
                    JessicaTask(

                        id =
                            obj.getLong(
                                "id"
                            ),

                        text =
                            obj.getString(
                                "text"
                            ),

                        status =
                            obj.getString(
                                "status"
                            ),

                        result =
                            obj.optString(
                                "result",
                                ""
                            ),

                        createdAt =
                            obj.getLong(
                                "createdAt"
                            )

                    )
                )

            }


            result

        } catch (e: Exception) {

            emptyList()

        }

    }


    fun updateTask(
        taskId: Long,
        status: String,
        result: String
    ) {

        val tasks =
            loadTasks()
                .toMutableList()


        val index =
            tasks.indexOfFirst {
                it.id == taskId
            }


        if (index == -1) {
            return
        }


        val oldTask =
            tasks[index]


        tasks[index] =
            oldTask.copy(
                status = status,
                result = result
            )


        saveTasks(
            tasks
        )

    }


    fun deleteTask(
        taskId: Long
    ) {

        val tasks =
            loadTasks()
                .filterNot {
                    it.id == taskId
                }


        saveTasks(
            tasks
        )

    }


    fun clearTasks() {

        context.deleteFile(
            fileName
        )

    }


    private fun saveTasks(
        tasks: List<JessicaTask>
    ) {

        val array =
            JSONArray()


        tasks.forEach { task ->

            val obj =
                JSONObject()


            obj.put(
                "id",
                task.id
            )


            obj.put(
                "text",
                task.text
            )


            obj.put(
                "status",
                task.status
            )


            obj.put(
                "result",
                task.result
            )


            obj.put(
                "createdAt",
                task.createdAt
            )


            array.put(
                obj
            )

        }


        context
            .openFileOutput(
                fileName,
                Context.MODE_PRIVATE
            )
            .use { output ->

                output.write(
                    array
                        .toString()
                        .toByteArray()
                )

            }

    }

}
