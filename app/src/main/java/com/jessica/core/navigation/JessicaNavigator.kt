package com.jessica.core.navigation


import androidx.compose.runtime.mutableStateOf


class JessicaNavigator {


    val currentPage =

        mutableStateOf(
            JessicaPage.HOME
        )


    fun navigateTo(

        page: JessicaPage

    ) {


        currentPage.value =
            page

    }


    fun backHome() {


        currentPage.value =
            JessicaPage.HOME

    }


}
