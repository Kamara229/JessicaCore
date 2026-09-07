package com.jessica.core.core


import android.content.Context
import androidx.compose.runtime.mutableStateOf
import com.jessica.core.modules.Block
import com.jessica.core.modules.BlockManager
import com.jessica.core.modules.BlockStorage
import com.jessica.core.modules.ReportStorage


class JessicaController(

    context: Context

) {


    private val blockStorage =
        BlockStorage(context)



    val reportStorage =
        ReportStorage(context)



    val blockManager =
        BlockManager()



    val blocks =
        mutableStateOf(
            emptyList<Block>()
        )



    val message =
        mutableStateOf(
            "Jessica Core v0.1 запущена"
        )



    fun loadBlocks() {


        val savedBlocks =
            blockStorage.loadBlocks()



        savedBlocks.forEach { block ->


            val exists =
                blockManager
                    .getBlocks()
                    .any {
                        it.id == block.id
                    }



            if (!exists) {

                blockManager.addBlock(
                    block
                )

            }

        }



        blocks.value =
            blockManager
                .getBlocks()
                .toList()

    }



    fun installBlock(
        block: Block
    ) {


        val exists =
            blockManager
                .getBlocks()
                .any {
                    it.id == block.id
                }



        if (exists) {


            message.value =
                "Блок ${block.name} уже установлен"


            return

        }



        blockManager.addBlock(
            block
        )


        blocks.value =
            blockManager
                .getBlocks()
                .toList()



        blockStorage.saveBlocks(
            blocks.value
        )


        message.value =
            "Блок ${block.name} установлен"

    }



}
