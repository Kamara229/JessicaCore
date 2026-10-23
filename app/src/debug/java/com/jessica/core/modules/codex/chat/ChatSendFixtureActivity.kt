package com.jessica.core.modules.codex.chat

import android.app.Activity
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.ScrollView
import java.util.concurrent.atomic.AtomicInteger

/** Disposable test UI; excluded from the signed release APK. */
class ChatSendFixtureActivity : Activity() {
    companion object {
        val clicks = AtomicInteger(0)
        @Volatile var sentText = ""
        @Volatile var conversationScrollY = 0
    }
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.setSoftInputMode(android.view.WindowManager.LayoutParams.SOFT_INPUT_STATE_ALWAYS_HIDDEN)
        clicks.set(0); sentText = ""; conversationScrollY = 0
        val density = resources.displayMetrics.density
        val editor = EditText(this).apply { setText("Проверка вложенной стрелки"); minLines = 2 }
        val send = FrameLayout(this).apply {
            isClickable = true
            isFocusable = true
            importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_YES
            addView(TextView(this@ChatSendFixtureActivity).apply {
                text = "↑"
                contentDescription = "Отправить сообщение"
                importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_YES
                gravity = Gravity.CENTER
            }, FrameLayout.LayoutParams(-1,-1))
            setOnClickListener { sentText = editor.text.toString(); clicks.incrementAndGet(); editor.setText("") }
        }
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            addView(editor, LinearLayout.LayoutParams(0,(96*density).toInt(),1f))
            addView(send, LinearLayout.LayoutParams((56*density).toInt(),(56*density).toInt()))
        }
        setContentView(LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(ScrollView(this@ChatSendFixtureActivity).apply {
                isFillViewport = true
                isVerticalScrollBarEnabled = true
                addView(TextView(this@ChatSendFixtureActivity).apply {
                    text = (1..110).joinToString("\n") { "Тестовая строка переписки номер $it" }
                    setPadding(12,12,12,12)
                })
                setOnScrollChangeListener { _, _, scrollY, _, _ ->
                    conversationScrollY = scrollY
                }
            }, LinearLayout.LayoutParams(-1,(400*density).toInt()))
            addView(row)
        })
    }
}
