package com.jessica.core.ui.codex

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.jessica.core.ui.language.uiText

/**
 * Non-destructive migration: keep legacy GitHub Actions autonomy available.
 * Phone mode never silently uses Render and never falsely claims to drive
 * another app automatically.
 */
@Composable
fun CodexScreen() {
    var phone by remember { mutableStateOf(true) }
    Column(modifier=Modifier.fillMaxSize()) {
        Row(
            horizontalArrangement=Arrangement.spacedBy(8.dp),
            modifier=Modifier.padding(horizontal=16.dp,vertical=4.dp)
        ) {
            FilterChip(
                selected=phone,
                onClick={phone=true},
                label={Text(uiText("На телефоне","On phone"))}
            )
            FilterChip(
                selected=!phone,
                onClick={phone=false},
                label={Text(uiText("GitHub Actions (старый)","GitHub Actions (legacy)"))}
            )
        }
        if(phone) PhoneCodexScreen() else LegacyCodexScreen()
    }
}
