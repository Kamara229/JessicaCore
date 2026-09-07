package com.jessica.core.ui.chat


import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Text

import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember

import androidx.compose.ui.res.painterResource


@Composable
fun MessageActions(

    onCopy: () -> Unit

) {


    val expanded =
        remember {
            mutableStateOf(false)
        }



    IconButton(

        onClick = {

            expanded.value =
                true

        }

    ) {


        Text(
            "⋮"
        )


    }



    DropdownMenu(

        expanded =
            expanded.value,


        onDismissRequest = {

            expanded.value =
                false

        }

    ) {


        DropdownMenuItem(

            text = {

                Text(
                    "Копировать"
                )

            },


            onClick = {

                expanded.value =
                    false

                onCopy()

            }

        )


    }

}
