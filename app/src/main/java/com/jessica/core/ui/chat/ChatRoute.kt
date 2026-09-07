package com.jessica.core.ui.chat

import androidx.compose.runtime.Composable
import androidx.lifecycle.viewmodel.compose.viewModel


/*
 * =========================================================
 * JESSICA CHAT ROUTE
 * =========================================================
 *
 * Navigation layer.
 *
 * Не содержит UI.
 * Не содержит бизнес-логику.
 *
 * Только связывает:
 *
 * ViewModel ↔ Screen
 *
 */


@Composable
fun ChatRoute(
    onBack: () -> Unit
) {


    val viewModel:
            ChatViewModel =
        viewModel()


    val state =
        viewModel.state


    ChatScreen(

        state =
            state.value,


        onInputChange = {

            viewModel.updateInput(
                it
            )

        },


        onSend = {

            viewModel.sendMessage()

        },


        onBack =
            onBack

    )

}
