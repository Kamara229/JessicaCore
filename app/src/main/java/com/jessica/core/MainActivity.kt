package com.jessica.core

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            JessicaScreen()
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
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

    ) { padding ->

        Column(
            modifier = Modifier
                .padding(padding)
                .padding(20.dp)
        ) {

            Text(
                text = message
            )

            Spacer(
                modifier = Modifier.height(20.dp)
            )

            Button(
                onClick = {
                    message =
                        "Ожидание подключения блоков..."
                }
            ) {

                Text(
                    "Добавить блок"
                )
            }
        }
    }
}
