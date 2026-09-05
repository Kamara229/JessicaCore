package com.jessica.core.modules

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject


data class MemoryItem(
    val key: String,
    val value: String,
    val timestamp: Long
)


class MemoryStorage(
    private val context: Context
) {

    private val fileName =
        "jessica_memory.json"


    fun saveMemory(
        key: String,
        value: String
    ) {

        val memory =
            loadMemory()
                .toMutableList()


        memory.removeAll {
            it.key == key
        }


        memory.add(
            MemoryItem(
                key = key,
                value = value,
                timestamp =
                    System.currentTimeMillis()
            )
        )


        saveAll(
            memory
        )

    }


    fun loadMemory(): List<MemoryItem> {

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
                mutableListOf<MemoryItem>()


            for (
                i in 0 until array.length()
            ) {

                val obj =
                    array.getJSONObject(i)


                result.add(

                    MemoryItem(

                        key =
                            obj.getString(
                                "key"
                            ),

                        value =
                            obj.getString(
                                "value"
                            ),

                        timestamp =
                            obj.getLong(
                                "timestamp"
                            )

                    )

                )

            }


            result

        } catch (e: Exception) {

            emptyList()

        }

    }


    fun getMemory(
        key: String
    ): MemoryItem? {

        return loadMemory()
            .find {
                it.key == key
            }

    }


    fun clearMemory() {

        context.deleteFile(
            fileName
        )

    }


    private fun saveAll(
        memory: List<MemoryItem>
    ) {

        val array =
            JSONArray()


        memory.forEach { item ->

            val obj =
                JSONObject()


            obj.put(
                "key",
                item.key
            )


            obj.put(
                "value",
                item.value
            )


            obj.put(
                "timestamp",
                item.timestamp
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
