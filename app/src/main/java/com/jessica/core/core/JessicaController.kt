package com.jessica.core.core


import android.content.Context
import com.jessica.core.BuildConfig

import androidx.compose.runtime.mutableStateOf

import com.jessica.core.modules.Block
import com.jessica.core.modules.BlockManager
import com.jessica.core.modules.BlockStorage
import com.jessica.core.modules.ReportStorage
import com.jessica.core.modules.JessicaAIEngine
import com.jessica.core.modules.chat.ChatMessage
import com.jessica.core.modules.chat.ChatAttachment
import com.jessica.core.modules.chat.ChatAttachmentStore
import com.jessica.core.modules.chat.DocumentExtractionClient



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

    private val attachmentExtractor = DocumentExtractionClient(ChatAttachmentStore(context))



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
            "Jessica Core v${BuildConfig.VERSION_NAME} запущена"
        )



    init {
        // Legacy blocks are loaded into Jessica automatically at startup.
        loadBlocks()
    }

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

        request: String,

        history: List<ChatMessage> = emptyList(),

        attachments: List<ChatAttachment> = emptyList()

    ): String {


        val task =

            request.trim()



        if (task.isBlank()) {


            return "Запрос пуст."


        }



        message.value =

            "Jessica выполняет запрос"



        val documents = try {
            attachmentExtractor.extract(attachments)
        } catch (error: Exception) {
            message.value = "Ошибка чтения вложений"
            return "Не удалось обработать вложение: ${error.message}"
        }

        val result =

            aiEngine.solve(
                task,
                history,
                documents
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
