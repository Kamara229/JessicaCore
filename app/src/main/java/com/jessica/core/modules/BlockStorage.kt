package com.jessica.core.modules

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject


class BlockStorage(
    private val context: Context
) {

    private val fileName =
        "jessica_blocks.json"


    fun saveBlocks(
        blocks: List<Block>
    ) {

        val array =
            JSONArray()


        blocks.forEach { block ->

            val obj =
                JSONObject()


            obj.put(
                "id",
                block.id
            )

            obj.put(
                "name",
                block.name
            )

            obj.put(
                "version",
                block.version
            )

            obj.put(
                "author",
                block.author
            )

            obj.put(
                "type",
                block.type
            )

            obj.put(
                "status",
                block.status
            )

            obj.put(
                "description",
                block.description
            )


            val capabilitiesArray =
                JSONArray()


            block.capabilities.forEach { capability ->

                capabilitiesArray.put(
                    capability
                )

            }


            obj.put(
                "capabilities",
                capabilitiesArray
            )


            array.put(
                obj
            )

        }


        context.openFileOutput(
            fileName,
            Context.MODE_PRIVATE
        ).use { output ->

            output.write(
                array
                    .toString()
                    .toByteArray()
            )

        }

    }


    fun loadBlocks(): List<Block> {

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


            val blocks =
                mutableListOf<Block>()


            for (i in 0 until array.length()) {

                val obj =
                    array.getJSONObject(i)


                val capabilities =
                    mutableListOf<String>()


                val capabilitiesArray =
                    obj.optJSONArray(
                        "capabilities"
                    )


                if (capabilitiesArray != null) {

                    for (
                        capabilityIndex
                        in 0 until capabilitiesArray.length()
                    ) {

                        capabilities.add(
                            capabilitiesArray.getString(
                                capabilityIndex
                            )
                        )

                    }

                }


                val block =
                    Block(

                        id =
                            obj.optString(
                                "id",
                                ""
                            ),

                        name =
                            obj.optString(
                                "name",
                                "Unknown"
                            ),

                        version =
                            obj.optString(
                                "version",
                                "0.1"
                            ),

                        author =
                            obj.optString(
                                "author",
                                "Unknown"
                            ),

                        type =
                            obj.optString(
                                "type",
                                "unknown"
                            ),

                        status =
                            obj.optString(
                                "status",
                                "ACTIVE"
                            ),

                        description =
                            obj.optString(
                                "description",
                                ""
                            ),

                        capabilities =
                            capabilities

                    )


                blocks.add(
                    block
                )

            }


            blocks

        } catch (e: Exception) {

            emptyList()

        }

    }

}
