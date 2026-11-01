package com.jessica.core.modules.codex.chat

import android.graphics.PixelFormat
import android.os.SystemClock
import android.view.Gravity
import android.view.WindowManager
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView

internal fun CodexChatCoordinator.showOverlay() {

        val launch = PhoneChatRuntime.launch ?: return
        val layout = LinearLayout(service).apply { orientation = LinearLayout.VERTICAL; setBackgroundColor(0xffeeeeee.toInt()) }
        statusLabel = TextView(service).apply {
            text = "Джессика: выбери чат"
            setTextColor(0xff222222.toInt())
            maxLines = 3
            maxWidth = (service.resources.displayMetrics.density * 245).toInt()
            setPadding(12, 6, 12, 6)
        }
        // Owner's Stop stays above all other controls, including diagnostics.
        layout.addView(emergencyStopButton())
        idlePulseButton = Button(service).apply {
            text = "Мини-прокрутка: выкл"
            setOnClickListener {
                if (!confirmed || PhoneChatRuntime.launch !== launch || launch.stopRequested) {
                    notice("Сначала выбери чат: экспериментальный жест доступен только в текущей сессии.")
                    return@setOnClickListener
                }
                idlePulseEnabled = !idlePulseEnabled
                lastIdlePulseAttemptAt = SystemClock.elapsedRealtime()
                text = if (idlePulseEnabled) "Мини-прокрутка: ВКЛ (180 с)" else "Мини-прокрутка: выкл"
                notice(if (idlePulseEnabled)
                    "Экспериментальный жест включён. Только при генерации ChatGPT и " +
                    "подтверждённой области диалога. Сам по себе от сна не защищает."
                else "Мини-прокрутка выключена. Удержание экрана — отдельный механизм.")
            }
        }
        layout.addView(idlePulseButton)
        layout.addView(statusLabel)
        confirmButton = Button(service).apply {
            text = "Чат выбран"
            setOnClickListener {
                val window = ChatWindowReader.read(service)
                if (window == null) { notice("Сначала открой чат в приложении ChatGPT."); return@setOnClickListener }
                try {
                    if (window.composer != null) {
                        if (launch.conversation.plan.isBlank()) launch.conversation.plan = window.content.takeLast(12000)
                        // Set first handoff BEFORE any composer mutation/send: one selected thread only.
                        // No navigation, no auto-opening a chat or altering user-entered drafts.
                        launch.message = com.jessica.core.modules.codex.relay.RelayProtocol.introductionFor(launch.conversation.plan)
                        launch.prompt = com.jessica.core.modules.codex.relay.RelayProtocol.wrap(launch.message, launch.replyId)
                        confirmed = true
                        reportPhase("chat_selected")
                        visibility = android.view.View.GONE
                        notice("Чат выбран. Джессика начинает переписку.")
                        observeConversationBanner(window.content)
                    } else {
                        // The banner alone must not stop or move the session.
                        observeConversationBanner(window.content)
                        notice("Не удаётся определить поле сообщения (обнаружено: ${window.editorCandidates}). " +
                            "Остаюсь в выбранном чате, новых сообщений не отправляю.")
                    }
                } finally { window.release() }
            }
        }
        layout.addView(confirmButton)
        calibrateButton = Button(service).apply {
            text = "Указать стрелку"
            visibility = android.view.View.GONE
            setOnClickListener { selectSendTarget(launch) }
        }
        layout.addView(calibrateButton)
        service.getSystemService(WindowManager::class.java).addView(layout, WindowManager.LayoutParams(
            WindowManager.LayoutParams.WRAP_CONTENT, WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL or WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON,
            PixelFormat.TRANSLUCENT).apply { gravity = Gravity.TOP or Gravity.END; y = 150 })
        overlay = layout
    
}
