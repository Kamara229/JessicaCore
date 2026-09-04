package com.jessica.core.modules

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject


class ReportStorage(
    private val context: Context
) {


    private val fileName =
        "jessica_reports.json"



    fun saveReports(
        reports: List<TestReport>
    ) {


        val array =
            JSONArray()


        reports.forEach {


            val obj =
                JSONObject()


            obj.put(
                "blockName",
                it.blockName
            )


            obj.put(
                "date",
                it.date
            )


            obj.put(
                "result",
                it.result
            )


            array.put(obj)

        }


        context.openFileOutput(
            fileName,
            Context.MODE_PRIVATE
        )
            .use {

                it.write(
                    array.toString()
                        .toByteArray()
                )

            }

    }



    fun loadReports(): List<TestReport> {


        return try {


            val text =
                context.openFileInput(fileName)
                    .bufferedReader()
                    .readText()


            val array =
                JSONArray(text)


            val result =
                mutableListOf<TestReport>()


            for(i in 0 until array.length()) {


                val obj =
                    array.getJSONObject(i)


                result.add(

                    TestReport(

                        obj.getString(
                            "blockName"
                        ),

                        obj.getString(
                            "date"
                        ),

                        obj.getString(
                            "result"
                        )

                    )

                )

            }


            result


        } catch(e: Exception) {


            emptyList()


        }


    }


}
