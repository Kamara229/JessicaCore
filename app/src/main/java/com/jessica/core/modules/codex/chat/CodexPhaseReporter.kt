package com.jessica.core.modules.codex.chat

import android.os.SystemClock
import kotlinx.coroutines.launch

internal fun CodexChatCoordinator.reportPhase(phase: String) {

        if (lastReportedPhase == phase) return
        val launch = PhoneChatRuntime.launch ?: return
        val state = run ?: return
        lastReportedPhase = phase
        val seconds = ((SystemClock.elapsedRealtime() - state.startedAt) / 1000L)
            .coerceIn(0L, 30_000L).toInt()
        scope.launch {
            try {
                PhoneChatTelemetry.send(launch.sessionId, phase, state.sends, seconds)
            } catch (_: Exception) {
                // Logging is strictly best-effort.
            }
        }
    
}
