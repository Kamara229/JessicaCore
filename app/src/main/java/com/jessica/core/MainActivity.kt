package com.jessica.core

import android.content.Context
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.jessica.core.modules.Block
import com.jessica.core.modules.BlockManager
import com.jessica.core.modules.BlockStorage
import com.jessica.core.ui.BlockScreen
import org.json.JSONObject


class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            JessicaScreen()
        }
    }
}


fun readBlockFromFile(
    context: Context,
    uri: Uri
): Block? {

    return try {

        val jsonText =
            context.contentResolver
                .openInputStream(uri)
                ?.bufferedReader()
                ?.use {
                    it.readText()
                }
                ?: return null


        val json =
            JSONObject(jsonText)


        Block(

            name =
            json.getString("name"),

            version =
            json.optString(
                "version",
                "0.1"
            ),

            status =
            json.optString(
                "status",
                "ACTIVE"
            )

        )


    } catch (e: Exception) {

        null

    }

}


@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun JessicaScreen() {


    val context =
        androidx.compose.ui.platform.LocalContext.current


    val storage =
        remember {
            BlockStorage(context)
        }


    val blockManager =
        remember {
            BlockManager()
        }


    var blocks by remember {

        mutableStateOf(
            emptyList<Block>()
        )

    }


    var message by remember {

        mutableStateOf(
            "Jessica Core v0.1 запущена"
        )

    }


    var showBlocks by remember {

        mutableStateOf(false)

    }


    LaunchedEffect(Unit) {


        val savedBlocks =
            storage.loadBlocks()


        savedBlocks.forEach {


            if (
                !blockManager
                    .getBlocks()
                    .contains(it)
            ) {

                blockManager.addBlock(it)

            }

        }


        blocks =
            blockManager
                .getBlocks()
                .toList()

    }



    val blockPicker =
        rememberLauncherForActivityResult(

            contract =
            ActivityResultContracts.OpenDocument()

        ) { uri ->


            if (uri != null) {


                val newBlock =
                    readBlockFromFile(
                        context,
                        uri
                    )


                if (newBlock == null) {


                    message =
                        "Ошибка чтения блока"


                } else {


                    if (
                        !blockManager
                            .getBlocks()
                            .contains(newBlock)
                    ) {


                        blockManager
                            .addBlock(newBlock)


                        blocks =
                            blockManager
                                .getBlocks()
                                .toList()


                        storage
                            .saveBlocks(blocks)


                        message =
                            "Блок ${newBlock.name} установлен"


                    } else {


                        message =
                            "Этот блок уже установлен"


                    }

                }

            }

        }



    Scaffold(

        topBar = {

            TopAppBar(

                title = {

                    Text(
                        "Jessica Core"
                    )

                }

            )

        }


    ) { padding ->



        Column(

            modifier =
            Modifier
                .padding(padding)
                .padding(20.dp)

        ) {



            Button(

                onClick = {

                    showBlocks =
                        !showBlocks

                }

            ) {


                Text(

                    if(showBlocks)
                        "Назад"
                    else
                        "Блоки"

                )


            }



            Spacer(

                modifier =
                Modifier.height(20.dp)

            )



            if(showBlocks) {


                BlockScreen(

                    blockManager =
                    blockManager,


                    onUpdate = {


                        blocks =
                            blockManager
                                .getBlocks()
                                .toList()


                        storage
                            .saveBlocks(blocks)

                    }

                )


            } else {



                Text(
                    text = message
                )


                Spacer(

                    modifier =
                    Modifier.height(20.dp)

                )


                Text(
                    text =
                    "Установленные блоки:"
                )



                if(blocks.isEmpty()) {


                    Text(
                        "Нет установленных блоков"
                    )


                }



                blocks.forEach {


                    Text(

                        text =
                        "${it.name} v${it.version} (${it.status})"

                    )


                }



                Spacer(

                    modifier =
                    Modifier.height(20.dp)

                )



                Button(

                    onClick = {


                        blockPicker.launch(

                            arrayOf(
                                "application/json",
                                "text/plain",
                                "*/*"
                            )

                        )


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
