package com.jessica.core.ui.chat


import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding

import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text

import androidx.compose.runtime.Composable

import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp


@Composable
fun ThinkingIndicator() {


    Row(

        modifier =
            Modifier.padding(
                horizontal = 12.dp,
                vertical = 8.dp
            ),

        horizontalArrangement =
            Arrangement.spacedBy(8.dp),

        verticalAlignment =
            Alignment.CenterVertically

    ) {


        CircularProgressIndicator(
            modifier =
                Modifier.padding(
                    2.dp
                )
        )


        Text(

            text = "Jessica думает…",

            style =
                MaterialTheme
                    .typography
                    .bodyMedium

        )


    }


}
