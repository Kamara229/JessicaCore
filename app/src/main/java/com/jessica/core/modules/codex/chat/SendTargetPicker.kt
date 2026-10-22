package com.jessica.core.modules.codex.chat

import android.content.Context
import android.graphics.PixelFormat
import android.view.Gravity
import android.view.MotionEvent
import android.view.WindowManager
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView

/** Captures one explicit owner point; the selection tap is never passed through to ChatGPT. */
class SendTargetPicker(private val context: Context) {
    private var view: FrameLayout? = null
    fun show(onPick: (Float,Float) -> Unit, onCancel: () -> Unit) {
        close()
        val layer = FrameLayout(context).apply { setBackgroundColor(0x22000000) }
        val instructions = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(0xffeeeeee.toInt())
            addView(TextView(context).apply {
                text = "Коснись центра синей стрелки отправки в ChatGPT. Джессика запомнит её для этой сессии."
                setTextColor(0xff222222.toInt()); setPadding(20,12,20,12)
            })
            addView(Button(context).apply { text = "Отмена"; setOnClickListener { close(); onCancel() } })
        }
        layer.addView(instructions, FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT,
            FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.TOP).apply { topMargin = (32 * context.resources.displayMetrics.density).toInt() })
        layer.setOnTouchListener { _, event ->
            if (event.actionMasked == MotionEvent.ACTION_UP) { val x = event.rawX; val y = event.rawY; close(); onPick(x,y) }
            true
        }
        context.getSystemService(WindowManager::class.java).addView(layer, WindowManager.LayoutParams(
            WindowManager.LayoutParams.MATCH_PARENT, WindowManager.LayoutParams.MATCH_PARENT,
            WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
            PixelFormat.TRANSLUCENT))
        view = layer
    }
    fun close() {
        view?.let { runCatching { context.getSystemService(WindowManager::class.java).removeView(it) } }
        view = null
    }
}
