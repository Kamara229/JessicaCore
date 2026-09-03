package com.jessica.core

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.jessica.core.modules.Block
import com.jessica.core.modules.BlockManager
import com.jessica.core.modules.BlockStorage

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

    val context = androidx.compose.ui.platform.LocalContext.current

val storage = remember {
    BlockStorage(context)
}

val blockManager = remember {
    BlockManager()
}

    var message by remember {
        mutableStateOf(
            "Jessica Core v0.1 запущена"
        )
    }

    var blocks by remember {
        mutableStateOf(
            blockManager.getBlocks()
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


            Text(
                text = "Установленные блоки:"
            )


            blocks.forEach {

                Text(
                    text =
                    "${it.name} v${it.version} (${it.status})"
                )

            }


            Spacer(
                modifier = Modifier.height(20.dp)
            )


            Button(

                onClick = {

                    val newBlock =
                        Block(
                            name = "Basic Analysis",
                            version = "0.1",
                            status = "ACTIVE"
                        )


                    blockManager.addBlock(newBlock)

blocks =
    blockManager.getBlocks()

storage.saveBlocks(blocks)


                    message =
                        "Добавлен новый блок"

                }

            ){

                Text(
                    "+ Добавить блок"
                )

            }

        }

    }

}
