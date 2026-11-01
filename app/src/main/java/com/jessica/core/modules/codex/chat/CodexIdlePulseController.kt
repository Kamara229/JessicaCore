package com.jessica.core.modules.codex.chat

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.app.KeyguardManager
import android.graphics.Path
import android.os.PowerManager
import android.os.SystemClock

/**
 * Opt-in, fail-closed experimental 180-second idle gesture. Never touches a
 * keyboard, navigation bar, side drawer, dialog, Send or Stop.
 *
 * The actual screen-on mechanism must be validated separately on the phone.
 */
internal fun CodexChatCoordinator.maybeIdlePulse(
    window: ChatWindow, launch: PhoneChatLaunch, state: ChatRunState
) {
    if (!idlePulseEnabled || !state.waiting || !window.generating || launch.stopRequested ||
        PhoneChatRuntime.launch !== launch || !confirmed || calibrating ||
        pendingInterpretation != null) return
    val now = SystemClock.elapsedRealtime()
    if (now - maxOf(lastIdlePulseAttemptAt, state.sendAt) < CodexIdlePulsePolicy.INTERVAL_MS)
        return

    val power = service.getSystemService(PowerManager::class.java)
    val locked = service.getSystemService(KeyguardManager::class.java).isKeyguardLocked
    if (!power.isInteractive || locked || window.composer == null) return
    // Accessibility trees are transient; a stale node must never end the whole session.
    val frame = runCatching { CodexIdlePulseTarget.find(window, launch.replyId,
        service.resources.displayMetrics.density) }.getOrNull()
    if (!CodexScrollSafetyPolicy.mayScrollAutomatically(idlePulseEnabled, frame != null)) return
    val target = frame ?: return
    val composer = window.composer.screenBox()
    val coords = CodexIdlePulsePolicy.coordinates(target, composer, service.resources.displayMetrics.density)
        ?: return
    if (!CodexIdlePulsePolicy.due(
            optedIn = idlePulseEnabled, selected = confirmed, waiting = state.waiting,
            generating = window.generating, screenInteractive = power.isInteractive,
            keyguardLocked = locked, eligibleTranscript = true,
            sendAt = state.sendAt, lastAttemptAt = lastIdlePulseAttemptAt, now = now
        )) return

    val up = Path().apply {
        moveTo(coords.x, coords.upStartY)
        lineTo(coords.x, coords.upEndY)
    }
    val down = Path().apply {
        moveTo(coords.x, coords.downStartY)
        lineTo(coords.x, coords.downEndY)
    }
    val gesture = GestureDescription.Builder()
        .addStroke(GestureDescription.StrokeDescription(up, 0, 200))
        .addStroke(GestureDescription.StrokeDescription(down, 350, 200))
        .build()

    // Attempt is accounted BEFORE dispatch: never retry even if Android
    // reports an uncertain outcome or cancellation.
    lastIdlePulseAttemptAt = now
    idlePulseUsed = true
    val started = try {
        service.dispatchGesture(gesture, object : AccessibilityService.GestureResultCallback() {
            override fun onCancelled(description: GestureDescription?) {
                if (PhoneChatRuntime.launch === launch && !launch.stopRequested) {
                    idlePulseEnabled = false
                    idlePulseButton?.text = "Мини-прокрутка: выкл"
                    notice("Жест отменён Android. Мини-прокрутка выключена, " +
                        "не повторяю действие автоматически.")
                }
            }
        }, null)
    } catch (_: RuntimeException) {
        false
    }
    if (!started) {
        idlePulseEnabled = false
        idlePulseButton?.text = "Мини-прокрутка: выкл"
        notice("Android не принял жест. Мини-прокрутка отключена.")
    }
}
