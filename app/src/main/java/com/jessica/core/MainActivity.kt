package com.jessica.core

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.material3.*
import androidx.compose.runtime.*

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {

            JessicaScreen()

        }
    }
}


@Composable
fun JessicaScreen() {

    var message by remember {
        mutableStateOf(
            "Jessica Core v0.1 запущена"
        )
    }


    Scaffold(

        topBar = {

            TopAppBar(
                title = {
                    Text("Jessica Core")
                }
            )

        }

    ){ padding ->


        Column(
            modifier = androidx.compose.ui.Modifier
                .padding(padding)
                .padding(20.dp)
        ){

            Text(
                text = message
            )


            Spacer(
                androidx.compose.ui.unit.dp
                    .let { androidx.compose.ui.Modifier.padding(it) }
            )


            Button(
                onClick = {

                    message =
                    "Ожидание подключения блоков..."

                }
            ){

                Text(
                    "Добавить блок"
                )

            }

        }

    }

}
