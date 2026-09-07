package com.jessica.core.ui


import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding

import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar

import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember

import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext

import com.jessica.core.core.JessicaController

import com.jessica.core.navigation.JessicaNavigator
import com.jessica.core.navigation.JessicaPage

import com.jessica.core.modules.blocks.BlockFileLoader
import com.jessica.core.modules.updater.UpdateViewModel

import com.jessica.core.ui.home.HomeScreen



@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun JessicaScreen() {


    val context =
        LocalContext.current



    val controller =
        remember {

            JessicaController(
                context
            )

        }



    val navigator =
        remember {

            JessicaNavigator()

        }



    val updateViewModel =
        remember {

            UpdateViewModel()

        }



    val updateState by
        updateViewModel.state
            .collectAsState()



    LaunchedEffect(Unit) {


        controller.loadBlocks()


    }



    val blockPicker =
        rememberLauncherForActivityResult(

            contract =
                ActivityResultContracts.OpenDocument()

        ) { uri ->


            if (uri != null) {


                val block =
                    BlockFileLoader.read(
                        context,
                        uri
                    )


                if (block != null) {


                    controller.installBlock(
                        block
                    )


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


            when (

                navigator.currentPage.value

            ) {



                JessicaPage.HOME -> {


                    HomeScreen(

                        message =
                            controller.message.value,


                        blockCount =
                            controller.blocks.value.size,


                        updateState =
                            updateState,


                        onUpdate = {


                            updateViewModel
                                .checkUpdate()


                        },


                        onBlocks = {


                            navigator.navigateTo(
                                JessicaPage.BLOCKS
                            )


                        },


                        onReports = {


                            navigator.navigateTo(
                                JessicaPage.REPORTS
                            )


                        },


                        onMemory = {


                            navigator.navigateTo(
                                JessicaPage.MEMORY
                            )


                        },


                        onTasks = {


                            navigator.navigateTo(
                                JessicaPage.TASKS
                            )


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
                            controller.blockManager,


                        reportStorage =
                            controller.reportStorage,


                        onUpdate = {


                            controller.loadBlocks()


                        },


                        onBack = {


                            navigator.backHome()


                        }


                    )


                }



                JessicaPage.REPORTS -> {


                    ReportScreen(

                        reportStorage =
                            controller.reportStorage,


                        onBack = {


                            navigator.backHome()


                        }


                    )


                }



                JessicaPage.MEMORY -> {


                    MemoryScreen(

                        onBack = {


                            navigator.backHome()


                        }

                    )


                }



                JessicaPage.TASKS -> {


                    TaskScreen(

                        onBack = {


                            navigator.backHome()


                        }

                    )


                }


            }


        }


    }


}
