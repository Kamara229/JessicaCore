package com.jessica.core.modules.codex.chat

/**
 * Pure state/cadence gate for the optional 3-minute idle gesture.
 *
 * A gesture cannot be a screen-awake guarantee: timeout may be shorter than
 * 180 seconds. The existing visible Codex overlay requests KEEP_SCREEN_ON.
 * The gesture is OFF per session until the owner explicitly opts in.
 */
internal object CodexIdlePulsePolicy {
    const val INTERVAL_MS = 180_000L

    fun due(
        optedIn: Boolean,
        selected: Boolean,
        waiting: Boolean,
        generating: Boolean,
        screenInteractive: Boolean,
        keyguardLocked: Boolean,
        eligibleTranscript: Boolean,
        sendAt: Long,
        lastAttemptAt: Long,
        now: Long
    ): Boolean {
        if (!optedIn || !selected || !waiting || !generating ||
            !screenInteractive || keyguardLocked || !eligibleTranscript) return false
        if (sendAt <= 0L || lastAttemptAt < 0L || now < sendAt || now < lastAttemptAt) return false
        return now - maxOf(sendAt, lastAttemptAt) >= INTERVAL_MS
    }

    /**
     * Two separate *tiny* strokes in the central scroll pane, with a return
     * stroke cancelling the visual displacement. Never inside the composer.
     */
    fun coordinates(target: UiBox, composer: UiBox, density: Float): CodexIdlePulseCoordinates? {
        if (!density.isFinite() || density <= 0f || target.width <= 0 || target.height <= 0) return null
        if (target.width < 250f * density || target.height < 250f * density ||
            target.bottom > composer.top - 12f * density) return null
        val midY = (target.top + target.bottom) / 2f
        val delta = 16f * density
        if (midY - delta < target.top + 70f * density ||
            midY + delta > target.bottom - 70f * density) return null
        return CodexIdlePulseCoordinates(
            x = target.left + target.width * .5f,
            upStartY = midY + delta,
            upEndY = midY - delta,
            downStartY = midY - delta,
            downEndY = midY + delta
        )
    }
}

internal data class CodexIdlePulseCoordinates(
    val x: Float,
    val upStartY: Float,
    val upEndY: Float,
    val downStartY: Float,
    val downEndY: Float
)
