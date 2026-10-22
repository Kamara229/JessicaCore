package com.jessica.core.modules.codex.chat

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.graphics.Path

/** One tap at the owner's selected composer-relative point. No retry after an uncertain outcome. */
internal fun dispatchSendTap(service: AccessibilityService, point: Pair<Float,Float>, cancelled: () -> Unit): Boolean {
    val path = Path().apply { moveTo(point.first,point.second) }
    val gesture = GestureDescription.Builder().addStroke(GestureDescription.StrokeDescription(path,0,60)).build()
    return service.dispatchGesture(gesture, object : AccessibilityService.GestureResultCallback() {
        override fun onCancelled(gestureDescription: GestureDescription?) { cancelled() }
    }, null)
}
