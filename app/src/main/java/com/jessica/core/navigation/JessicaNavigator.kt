package com.jessica.core.navigation


import androidx.compose.runtime.mutableStateOf



enum class JessicaPage {

    HOME,

    BLOCKS,

    REPORTS,

    MEMORY,

    TASKS

}



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
