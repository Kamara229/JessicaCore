package com.jessica.core


import android.os.Bundle

import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent

import com.jessica.core.ui.JessicaScreen



class MainActivity : ComponentActivity() {


    override fun onCreate(
        savedInstanceState: Bundle?
    ) {


        super.onCreate(
            savedInstanceState
        )


        setContent {


            JessicaScreen()


        }

    }


}
