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
import com.jessica.core.modules.ReportStorage
import com.jessica.core.ui.BlockScreen
import com.jessica.core.ui.ReportScreen
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


        val text =

            context.contentResolver
                .openInputStream(uri)
                ?.bufferedReader()
                ?.use {

                    it.readText()

                }
                ?: return null



        val json =
            JSONObject(text)



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



    val reportStorage =
        remember {

            ReportStorage(context)

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





    var showReports by remember {


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


                val block =

                    readBlockFromFile(
                        context,
                        uri
                    )



                if (block != null) {



                    if (
                        !blockManager
                            .getBlocks()
                            .contains(block)
                    ) {


                        blockManager.addBlock(block)



                        blocks =

                            blockManager
                                .getBlocks()
                                .toList()



                        storage.saveBlocks(blocks)



                        message =
                            "Блок ${block.name} установлен"



                    } else {


                        message =
                            "Блок уже установлен"


                    }



                } else {


                    message =
                        "Ошибка чтения блока"



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



            Row {



                Button(

                    onClick = {


                        showBlocks = true

                        showReports = false


                    }

                ) {


                    Text(
                        "Блоки"
                    )


                }



                Spacer(

                    modifier =
                    Modifier.width(10.dp)

                )



                Button(

                    onClick = {


                        showReports = true

                        showBlocks = false


                    }

                ) {


                    Text(
                        "Отчёты"
                    )


                }



            }






            Spacer(

                modifier =
                Modifier.height(20.dp)

            )






            when {


                showBlocks -> {



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



                }





                showReports -> {



                    ReportScreen(

                        reportStorage =
                        reportStorage

                    )


                }






                else -> {




                    Text(
                        message
                    )



                    Spacer(

                        modifier =
                        Modifier.height(20.dp)

                    )



                    Text(
                        "Установлено блоков: ${blocks.size}"
                    )



                    Spacer(

                        modifier =
                        Modifier.height(20.dp)

                    )



                    Button(


                        onClick = {


                            blockPicker.launch(

                                arrayOf(
                                    "application/json",
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




}
