package com.jessica.core.modules

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject


data class JessicaEvent(
    val type: String,
    val message: String,
    val timestamp: Long
)


class EventStorage(
    private val context: Context
) {

    private val fileName =
        "jessica_events.json"


    fun saveEvent(
        type: String,
        message: String
    ) {

        val events =
            loadEvents()
                .toMutableList()


        events.add(
            JessicaEvent(
                type = type,
                message = message,
                timestamp = System.currentTimeMillis()
            )
        )


        saveAll(
            events
        )

    }


    fun loadEvents(): List<JessicaEvent> {

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
                mutableListOf<JessicaEvent>()


            for (
                i in 0 until array.length()
            ) {

                val obj =
                    array.getJSONObject(i)


                result.add(
                    JessicaEvent(

                        type =
                            obj.getString(
                                "type"
                            ),

                        message =
                            obj.getString(
                                "message"
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


    fun clearEvents() {

        context.deleteFile(
            fileName
        )

    }


    private fun saveAll(
        events: List<JessicaEvent>
    ) {

        val array =
            JSONArray()


        events.forEach { event ->

            val obj =
                JSONObject()


            obj.put(
                "type",
                event.type
            )


            obj.put(
                "message",
                event.message
            )


            obj.put(
                "timestamp",
                event.timestamp
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
