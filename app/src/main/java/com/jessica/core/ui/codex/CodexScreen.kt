package com.jessica.core.ui.codex

import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
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

/** Ordinary chat is explicit on-device automation; legacy modes remain available. */
@Composable
fun CodexScreen() {
    var mode by remember { mutableStateOf("ordinary") }
    Column(modifier=Modifier.fillMaxSize()) {
        Row(
            modifier=Modifier.horizontalScroll(rememberScrollState())
                .padding(horizontal=16.dp,vertical=4.dp),
            horizontalArrangement=Arrangement.spacedBy(8.dp)
        ) {
            FilterChip(
                selected=mode=="ordinary", onClick={mode="ordinary"},
                label={Text(uiText("Обычный чат","Ordinary chat"))}
            )
            FilterChip(
                selected=mode=="day", onClick={mode="day"},
                label={Text(uiText("☀ Дневной режим","☀ Day shift"))}
            )
            FilterChip(
                selected=mode=="phone", onClick={mode="phone"},
                label={Text(uiText("Через ChatGPT","Via ChatGPT"))}
            )
            FilterChip(
                selected=mode=="legacy", onClick={mode="legacy"},
                label={Text(uiText("Codex API (платный)","Codex API (paid)"))}
            )
        }
        when(mode) {
            "ordinary" -> OrdinaryChatScreen()
            "day" -> DayCodexScreen()
            "phone" -> PhoneCodexScreen()
            else -> LegacyCodexScreen()
        }
    }
}
