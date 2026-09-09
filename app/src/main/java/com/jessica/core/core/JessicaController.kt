package com.jessica.core.core


import android.content.Context

import androidx.compose.runtime.mutableStateOf

import com.jessica.core.modules.Block
import com.jessica.core.modules.BlockManager
import com.jessica.core.modules.BlockStorage
import com.jessica.core.modules.ReportStorage



/*
 * =========================================================
 * JESSICA CONTROLLER
 * =========================================================
 *
 * Центральное ядро Android-клиента Jessica.
 *
 * Связывает:
 *
 * - UI
 * - Chat
 * - Blocks
 * - Reports
 * - будущий AI Engine
 *
 * ВАЖНО:
 *
 * UI не должен напрямую обращаться
 * к Planner / API / Tool Registry.
 *
 * Все запросы проходят через
 * JessicaController.
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
     * Главная точка входа сообщений из ChatViewModel.
     *
     * Сейчас здесь безопасная промежуточная реализация.
     *
     * Позже цепочка станет:
     *
     * ChatViewModel
     *      ↓
     * JessicaController
     *      ↓
     * Planner
     *      ↓
     * TaskRunner
     *      ↓
     * Tool Registry
     *      ↓
     * Validator
     *      ↓
     * Answer Composer
     *
     * =====================================================
     */


    suspend fun executeChatMessage(

        request: String

    ): String {


        val text =

            request.trim()



        if (text.isBlank()) {


            return "Запрос пуст."


        }



        message.value =

            "Jessica обрабатывает запрос"



        /*
         * =================================================
         * ВРЕМЕННАЯ ТОЧКА ПОДКЛЮЧЕНИЯ
         * =================================================
         *
         * Именно этот участок следующим шагом
         * подключим к реальному Jessica backend.
         *
         * Никакой AI-логики в ChatViewModel
         * больше размещать не потребуется.
         */


        val result =

            "Запрос получен ядром Jessica: $text"



        message.value =

            "Jessica готова к работе"



        return result


    }


}
