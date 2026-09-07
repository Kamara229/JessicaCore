package com.jessica.core.modules.updater


sealed class UpdateState {


    object Idle : UpdateState()


    object Checking : UpdateState()


    data class Downloading(

        val progress: Int

    ) : UpdateState()



    data class Completed(

        val version: String

    ) : UpdateState()



    data class Error(

        val message: String

    ) : UpdateState()

}
