package com.jessica.core.modules.codex.chat

import android.graphics.PixelFormat
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
        layout.addView(statusLabel)
        confirmButton = Button(service).apply {
            text = "Чат выбран"
            setOnClickListener {
                val window = ChatWindowReader.read(service)
                if (window == null) { notice("Сначала открой чат в приложении ChatGPT."); return@setOnClickListener }
                try {
                    if (window.composer != null) {
                        if (launch.conversation.plan.isBlank()) launch.conversation.plan = window.content.takeLast(12000)
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
