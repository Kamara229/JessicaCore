package com.jessica.core.ui


import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.platform.LocalContext
import androidx.compose.runtime.collectAsState

import com.jessica.core.core.JessicaController
import com.jessica.core.navigation.JessicaNavigator
import com.jessica.core.navigation.JessicaPage
import com.jessica.core.modules.updater.UpdateViewModel

import androidx.lifecycle.viewmodel.compose.viewModel



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
        viewModel<UpdateViewModel>()



    val updateState by
        updateViewModel.state
            .collectAsState()



    when (
        navigator.currentPage.value
    ) {


        JessicaPage.HOME -> {


            HomeScreenPlaceholder()


        }


        JessicaPage.BLOCKS -> {


            HomeScreenPlaceholder()


        }


        JessicaPage.REPORTS -> {


            HomeScreenPlaceholder()


        }


        JessicaPage.MEMORY -> {


            HomeScreenPlaceholder()


        }


        JessicaPage.TASKS -> {


            HomeScreenPlaceholder()


        }


    }


}



@Composable
private fun HomeScreenPlaceholder() {


}
