package com.jessica.core.modules.blocks


import android.content.Context
import android.net.Uri
import com.jessica.core.modules.Block
import org.json.JSONObject



object BlockFileLoader {



    fun read(

        context: Context,

        uri: Uri

    ): Block? {


        return try {


            val text =

                context
                    .contentResolver
                    .openInputStream(uri)
                    ?.bufferedReader()
                    ?.use {

                        it.readText()

                    }
                    ?: return null



            val json =

                JSONObject(text)



            val capabilitiesJson =

                json.optJSONArray(
                    "capabilities"
                )



            val capabilities =

                mutableListOf<String>()



            if (capabilitiesJson != null) {


                for (

                    i in 0 until capabilitiesJson.length()

                ) {


                    capabilities.add(

                        capabilitiesJson
                            .getString(i)

                    )

                }

            }



            Block(


                id =

                    json.optString(
                        "id",
                        ""
                    ),



                name =

                    json.getString(
                        "name"
                    ),



                version =

                    json.optString(
                        "version",
                        "0.1"
                    ),



                author =

                    json.optString(
                        "author",
                        "Unknown"
                    ),



                type =

                    json.optString(
                        "type",
                        "unknown"
                    ),



                status =

                    json.optString(
                        "status",
                        "ACTIVE"
                    ),



                description =

                    json.optString(
                        "description",
                        ""
                    ),



                capabilities =

                    capabilities

            )


        } catch (

            e: Exception

        ) {


            null

        }


    }


}
