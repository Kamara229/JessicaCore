package com.jessica.core.core


import android.content.Context

import androidx.compose.runtime.mutableStateOf

import com.jessica.core.modules.Block
import com.jessica.core.modules.BlockManager
import com.jessica.core.modules.BlockStorage
import com.jessica.core.modules.ReportStorage
import com.jessica.core.modules.JessicaAIEngine



/*
 * =========================================================
 * JESSICA CONTROLLER
 * =========================================================
 *
 * Центральный координатор Android-клиента Jessica.
 *
 * Связывает:
 *
 * - UI
 * - Chat
 * - Blocks
 * - Reports
 * - JessicaAIEngine
 *
 *
 * Основная цепочка чата:
 *
 * ChatViewModel
 *      ↓
 * JessicaController
 *      ↓
 * JessicaAIEngine
 *      ↓
 * /api/solve
 *      ↓
 * Jessica Core Backend
 *
 *
 * JessicaController НЕ содержит:
 *
 * - HTTP-реализацию
 * - Planner
 * - TaskRunner
 * - Tool Registry
 * - Validator
 * - Answer Composer
 *
 * =========================================================
 */


class JessicaController(

    context: Context

) {



    /*
     * =====================================================
     * STORAGE
     * =====================================================
     */


    private val blockStorage =

        BlockStorage(
            context
        )



    val reportStorage =

        ReportStorage(
            context
        )



    /*
     * =====================================================
     * BLOCK MANAGER
     * =====================================================
     */


    val blockManager =

        BlockManager()



    /*
     * =====================================================
     * AI ENGINE
     * =====================================================
     *
     * JessicaAIEngine отвечает за соединение
     * Android-приложения с backend Jessica Core.
     *
     * =====================================================
     */


    private val aiEngine =

        JessicaAIEngine()



    /*
     * =====================================================
     * UI STATE
     * =====================================================
     */


    val blocks =

        mutableStateOf(
            emptyList<Block>()
        )



    val message =

        mutableStateOf(
            "Jessica Core v0.1 запущена"
        )



    /*
     * =====================================================
     * LOAD BLOCKS
     * =====================================================
     */


    fun loadBlocks() {


        val savedBlocks =

            blockStorage
                .loadBlocks()



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



    /*
     * =====================================================
     * INSTALL BLOCK
     * =====================================================
     */


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



    /*
     * =====================================================
     * CHAT ENTRY POINT
     * =====================================================
     *
     * Единая точка входа для сообщений чата.
     *
     * Цепочка:
     *
     * ChatViewModel
     *      ↓
     * JessicaController
     *      ↓
     * JessicaAIEngine
     *      ↓
     * POST /api/solve
     *      ↓
     * Jessica Core
     *
     * =====================================================
     */


    suspend fun executeChatMessage(

        request: String

    ): String {


        val task =

            request.trim()



        if (task.isBlank()) {


            return "Запрос пуст."


        }



        message.value =

            "Jessica выполняет запрос"



        val result =

            aiEngine.solve(
                task
            )



        message.value =

            if (result.success) {


                "Jessica готова к работе"


            } else {


                "Ошибка выполнения запроса"


            }



        return result.text


    }


}
