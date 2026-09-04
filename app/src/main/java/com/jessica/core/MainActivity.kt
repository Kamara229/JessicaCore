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
import com.jessica.core.ui.BlockScreen


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


    var blocks by remember {
        mutableStateOf(
            emptyList<Block>()
        )
    }


    LaunchedEffect(Unit) {

        val savedBlocks = storage.loadBlocks()

        savedBlocks.forEach {

            if (!blockManager.getBlocks().contains(it)) {
                blockManager.addBlock(it)
            }

        }

        blocks = blockManager.getBlocks()

    }


    var message by remember {
        mutableStateOf(
            "Jessica Core v0.1 запущена"
        )
    }


    var showBlocks by remember {
        mutableStateOf(false)
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


            Button(

                onClick = {

                    showBlocks = !showBlocks

                }

            ) {

                Text(

                    if (showBlocks)
                        "Назад"
                    else
                        "Блоки"

                )

            }


            Spacer(

                modifier = Modifier.height(20.dp)

            )


            if (showBlocks) {


                BlockScreen(

                    blockManager = blockManager,

                    onUpdate = {

                        blocks =
                            blockManager.getBlocks()

                        storage.saveBlocks(blocks)

                    }

                )


            } else {


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


                        if (!blockManager.getBlocks().contains(newBlock)) {


                            blockManager.addBlock(newBlock)


                            blocks =
                                blockManager.getBlocks()


                            storage.saveBlocks(blocks)


                            message =
                                "Добавлен новый блок"


                        } else {


                            message =
                                "Такой блок уже установлен"

                        }


                    }

                ) {


                    Text(

                        "+ Добавить блок"

                    )


                }


            }


        }


    }


}
