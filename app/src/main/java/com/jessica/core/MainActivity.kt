package com.jessica.core

import android.content.Context
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.jessica.core.modules.Block
import com.jessica.core.modules.BlockManager
import com.jessica.core.modules.BlockStorage
import com.jessica.core.modules.ReportStorage
import com.jessica.core.ui.BlockScreen
import com.jessica.core.ui.MemoryScreen
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


enum class JessicaPage {

    HOME,
    BLOCKS,
    REPORTS,
    MEMORY

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


        val capabilitiesJson =
            json.optJSONArray(
                "capabilities"
            )


        val capabilities =
            mutableListOf<String>()


        if (capabilitiesJson != null) {

            for (
                i in 0 until capabilitiesJson.length()
            ) {

                capabilities.add(
                    capabilitiesJson.getString(i)
                )

            }

        }


        Block(

            id =
                json.optString(
                    "id",
                    ""
                ),

            name =
                json.getString(
                    "name"
                ),

            version =
                json.optString(
                    "version",
                    "0.1"
                ),

            author =
                json.optString(
                    "author",
                    "Unknown"
                ),

            type =
                json.optString(
                    "type",
                    "unknown"
                ),

            status =
                json.optString(
                    "status",
                    "ACTIVE"
                ),

            description =
                json.optString(
                    "description",
                    ""
                ),

            capabilities =
                capabilities

        )

    } catch (e: Exception) {

        null

    }

}


@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun JessicaScreen() {

    val context =
        LocalContext.current


    val blockStorage =
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


    var currentPage by remember {

        mutableStateOf(
            JessicaPage.HOME
        )

    }


    LaunchedEffect(Unit) {

        val savedBlocks =
            blockStorage.loadBlocks()


        savedBlocks.forEach { block ->

            val alreadyInstalled =
                blockManager
                    .getBlocks()
                    .any {
                        it.id == block.id
                    }


            if (!alreadyInstalled) {

                blockManager.addBlock(
                    block
                )

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

                    if (block.id.isBlank()) {

                        message =
                            "Ошибка: у блока отсутствует ID"

                    } else {

                        val alreadyInstalled =
                            blockManager
                                .getBlocks()
                                .any {
                                    it.id == block.id
                                }


                        if (!alreadyInstalled) {

                            blockManager.addBlock(
                                block
                            )


                            blocks =
                                blockManager
                                    .getBlocks()
                                    .toList()


                            blockStorage.saveBlocks(
                                blocks
                            )


                            message =
                                "Блок ${block.name} установлен"

                        } else {

                            message =
                                "Блок ${block.name} уже установлен"

                        }

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


        Box(

            modifier =
                Modifier
                    .padding(padding)
                    .fillMaxSize()

        ) {


            when (currentPage) {


                JessicaPage.HOME -> {

                    HomeScreen(

                        message =
                            message,

                        blockCount =
                            blocks.size,

                        onBlocks = {

                            currentPage =
                                JessicaPage.BLOCKS

                        },

                        onReports = {

                            currentPage =
                                JessicaPage.REPORTS

                        },

                        onMemory = {

                            currentPage =
                                JessicaPage.MEMORY

                        },

                        onAddBlock = {

                            blockPicker.launch(

                                arrayOf(
                                    "application/json",
                                    "*/*"
                                )

                            )

                        }

                    )

                }


                JessicaPage.BLOCKS -> {

                    BlockScreen(

                        blockManager =
                            blockManager,

                        reportStorage =
                            reportStorage,

                        onUpdate = {

                            blocks =
                                blockManager
                                    .getBlocks()
                                    .toList()


                            blockStorage.saveBlocks(
                                blocks
                            )

                        },

                        onBack = {

                            currentPage =
                                JessicaPage.HOME

                        }

                    )

                }


                JessicaPage.REPORTS -> {

                    ReportScreen(

                        reportStorage =
                            reportStorage,

                        onBack = {

                            currentPage =
                                JessicaPage.HOME

                        }

                    )

                }


                JessicaPage.MEMORY -> {

                    MemoryScreen(

                        onBack = {

                            currentPage =
                                JessicaPage.HOME

                        }

                    )

                }

            }

        }

    }

}


@Composable
fun HomeScreen(
    message: String,
    blockCount: Int,
    onBlocks: () -> Unit,
    onReports: () -> Unit,
    onMemory: () -> Unit,
    onAddBlock: () -> Unit
) {

    LazyColumn(

        modifier =
            Modifier
                .fillMaxSize()
                .padding(20.dp),

        verticalArrangement =
            Arrangement.spacedBy(10.dp)

    ) {


        item {

            Text(
                text = message
            )

        }


        item {

            Text(
                text =
                    "Установлено блоков: $blockCount"
            )

        }


        item {

            Spacer(
                modifier =
                    Modifier.height(10.dp)
            )

        }


        item {

            Button(
                onClick = onBlocks,
                modifier =
                    Modifier.fillMaxWidth()
            ) {

                Text(
                    "Блоки"
                )

            }

        }


        item {

            Button(
                onClick = onReports,
                modifier =
                    Modifier.fillMaxWidth()
            ) {

                Text(
                    "Отчёты"
                )

            }

        }


        item {

            Button(
                onClick = onMemory,
                modifier =
                    Modifier.fillMaxWidth()
            ) {

                Text(
                    "Память"
                )

            }

        }


        item {

            Spacer(
                modifier =
                    Modifier.height(10.dp)
            )

        }


        item {

            Button(
                onClick = onAddBlock,
                modifier =
                    Modifier.fillMaxWidth()
            ) {

                Text(
                    "+ Добавить блок"
                )

            }

        }


        item {

            Spacer(
                modifier =
                    Modifier.height(30.dp)
            )

        }

    }

}
