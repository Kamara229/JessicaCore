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
        layout.addView(statusLabel)
        confirmButton = Button(service).apply {
            text = "Чат выбран"
            setOnClickListener {
                val window = ChatWindowReader.read(service)
                if (window == null) { notice("Сначала открой чат в приложении ChatGPT."); return@setOnClickListener }
                try {
                    if (ChatUiPolicy.conversationAtLimit(window.content)) {
                        finish("В этом чате ChatGPT достигнут предел длины. " +
                            "Джессика не отправит сообщение и не переключится в другой чат без твоего выбора.")
                    } else if (window.composer != null) {
                        if (launch.conversation.plan.isBlank()) launch.conversation.plan = window.content.takeLast(12000)
                        confirmed = true
                        reportPhase("chat_selected")
                        visibility = android.view.View.GONE
                        notice("Чат выбран. Джессика начинает переписку.")
                    } else notice("Не удаётся определить поле сообщения (обнаружено: ${window.editorCandidates}). " +
                        "Открой нужную переписку ChatGPT; отправка не выполнялась.")
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
        layout.addView(emergencyStopButton())
        service.getSystemService(WindowManager::class.java).addView(layout, WindowManager.LayoutParams(
            WindowManager.LayoutParams.WRAP_CONTENT, WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL or WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON,
            PixelFormat.TRANSLUCENT).apply { gravity = Gravity.TOP or Gravity.END; y = 150 })
        overlay = layout
    
}
