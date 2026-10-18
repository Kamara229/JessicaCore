package com.jessica.core


import android.os.Bundle

import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent

import com.jessica.core.ui.JessicaScreen
import com.jessica.core.ui.theme.JessicaTheme



class MainActivity : ComponentActivity() {


    override fun onCreate(
        savedInstanceState: Bundle?
    ) {


        super.onCreate(
            savedInstanceState
        )


        setContent {


            JessicaTheme {
                JessicaScreen()
            }


        }

    }


}
