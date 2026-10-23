package com.jessica.core.modules.codex.chat

import android.view.MotionEvent
import android.widget.Button

/**
 * Dedicated emergency-stop control: touch-down stops Codex before ACTION_UP.
 * Also supports accessibility ACTION_CLICK. It is intentionally independent
 * of the ChatGPT conversation and never dispatches touch/scroll gestures.
 */
internal fun CodexChatCoordinator.emergencyStopButton(): Button =
    Button(service).apply {
        text = "■ СТОП КОДЕКС"
        setTextColor(0xffffffff.toInt())
        setBackgroundColor(0xffae1616.toInt())
        minimumHeight = (56f * service.resources.displayMetrics.density).toInt()
        isClickable = true
        isFocusable = true
        setOnTouchListener { _, event ->
            if (event.actionMasked == MotionEvent.ACTION_DOWN) {
                PhoneChatRuntime.launch?.requestStop()
                stopByUser("Кодекс немедленно остановлен пользователем.")
            }
            true // Consume release; cannot bubble into ChatGPT underneath.
        }
        setOnClickListener { stopByUser("Кодекс остановлен пользователем.") }
    }
