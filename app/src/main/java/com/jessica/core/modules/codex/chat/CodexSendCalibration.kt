package com.jessica.core.modules.codex.chat

import android.app.KeyguardManager

internal fun CodexChatCoordinator.tapSelectedSend(point: Pair<Float,Float>, launch: PhoneChatLaunch): Boolean {


        return dispatchSendTap(service, point) {
            if (PhoneChatRuntime.launch === launch) stopByUser("Android прервал нажатие. Проверь чат; повторной отправки не будет.")
        }
    

}
internal fun CodexChatCoordinator.selectSendTarget(launch: PhoneChatLaunch) {

        if (PhoneChatRuntime.launch !== launch || !confirmed || !composed || run?.waiting != false) return
        calibrating = true
        try {
            sendPicker.show(onPick = { x,y ->
                calibrating = false
                val window = ChatWindowReader.read(service, launch.prompt)
                try {
                    val editor = window?.composer
                    calibratedSend = if (PhoneChatRuntime.launch === launch && window != null && editor != null && editor.text?.toString() == launch.prompt && !window.generating &&
                        !service.getSystemService(KeyguardManager::class.java).isKeyguardLocked)
                        CalibratedSendTarget.select(window.root.screenBox(),editor.screenBox(),x,y,service.resources.displayMetrics.density) else null
                    notice(if (calibratedSend != null) "Стрелка выбрана. Отправляю подготовленное сообщение."
                        else "Точка не подходит. Нажми «Указать стрелку» и коснись синей стрелки справа от сообщения.")
                } finally { window?.release() }
            }, onCancel = { calibrating = false; notice("Выбор стрелки отменён. Сообщение не отправлено.") })
        } catch (e: Exception) { calibrating = false; finish("Не удалось показать выбор стрелки: ${e.message}") }
    
}
