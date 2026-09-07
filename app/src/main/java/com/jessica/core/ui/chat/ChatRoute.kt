package com.jessica.core.ui.chat


import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue


/*
 * =========================================================
 * JESSICA CHAT ROUTE
 * =========================================================
 *
 * Navigation layer.
 *
 * Отвечает только за связь:
 *
 * ChatViewModel → ChatScreen
 *
 * Не содержит:
 * - UI логику
 * - бизнес логику
 * - работу с памятью
 * - работу с блоками
 *
 * =========================================================
 */


@Composable
fun ChatRoute(

    viewModel: ChatViewModel,

    onBack: () -> Unit

) {


    val state by
        viewModel.state.collectAsState()



    ChatScreen(

        state = state,


        onInputChange = {

            viewModel.updateInput(
                it
            )

        },


        onSend = {

            viewModel.sendMessage()

        },


        onBack = {

            onBack()

        }

    )

}
