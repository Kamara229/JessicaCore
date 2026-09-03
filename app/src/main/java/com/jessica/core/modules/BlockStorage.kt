package com.jessica.core.modules

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

class BlockStorage(
    private val context: Context
) {

    private val prefs =
        context.getSharedPreferences(
            "jessica_blocks",
            Context.MODE_PRIVATE
        )


    fun saveBlocks(
        blocks: List<Block>
    ) {

        val array = JSONArray()

        blocks.forEach {

            val obj = JSONObject()

            obj.put("name", it.name)
            obj.put("version", it.version)
            obj.put("status", it.status)

            array.put(obj)
        }


        prefs.edit()
            .putString(
                "blocks",
                array.toString()
            )
            .apply()
    }


    fun loadBlocks(): List<Block> {

        val result =
            mutableListOf<Block>()


        val data =
            prefs.getString(
                "blocks",
                null
            )


        if (data != null) {

            val array =
                JSONArray(data)


            for (i in 0 until array.length()) {

                val obj =
                    array.getJSONObject(i)


                result.add(

                    Block(
                        name =
                        obj.getString("name"),

                        version =
                        obj.getString("version"),

                        status =
                        obj.getString("status")
                    )

                )
            }
        }


        return result
    }
}
