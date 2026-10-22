package com.jessica.core.modules.codex.chat

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/** Pure selection policy for ChatGPT's editable composer.
 *
 * ChatGPT may expose both a virtual parent and a native editable child, or
 * multiple editable nodes while its keyboard is displayed. Never require
 * exactly one editable node in the entire accessibility tree.
 *
 * We do NOT infer the send button from coordinates here.
 */
data class ComposerCandidate(
    val index: Int,
    val box: UiBox,
    val focused: Boolean,
    val editable: Boolean,
    val setText: Boolean,
    // Only true for one node whose actual editable text exactly matches this session's prepared draft.
    val matchesPrepared: Boolean = false
)

object ChatComposerPolicy {
    private fun nearSameField(a: UiBox, b: UiBox, density: Float): Boolean {
        val margin = max(10f, 12f * density)
        val xOverlap = (min(a.right, b.right) - max(a.left, b.left)).coerceAtLeast(0)
        val yOverlap = (min(a.bottom, b.bottom) - max(a.top, b.top)).coerceAtLeast(0)
        val w = max(1, min(a.width, b.width))
        val h = max(1, min(a.height, b.height))
        return (xOverlap.toFloat() / w >= .85f &&
            yOverlap.toFloat() / h >= .72f) ||
            (abs(a.left - b.left) <= margin && abs(a.right - b.right) <= margin &&
                abs(a.bottom - b.bottom) <= margin)
    }

    fun choose(candidates: List<ComposerCandidate>, frame: UiBox, density: Float): Int? {
        if (frame.width < 1 || frame.height < 1 || !density.isFinite() || density <= 0f) return null
        // ChatGPT can change the editable node's reported bounds after ACTION_SET_TEXT
        // expands a long draft. Recover ONLY an exact, unique, visible editor
        // matching this session's prepared text. Never guess an empty editor.
        val prepared = candidates.filter { it.matchesPrepared && (it.editable || it.setText) }
        if (prepared.size == 1) return prepared.single().index
        if (prepared.size > 1) {
            // ChatGPT can expose both an editable wrapper and its native child,
            // containing identical draft text. They are ONE field, not two.
            // Do not select anything if the same draft appears in distinct fields.
            if (!prepared.all { nearSameField(prepared.first().box, it.box, density) }) return null
            return prepared.maxWith(compareBy<ComposerCandidate> { it.setText }
                .thenBy { it.editable }.thenBy { it.focused }.thenBy { it.index }).index
        }
        val minimumWidth = max(72f * density, frame.width * .25f)
        val eligible = candidates.filter {
            (it.editable || it.setText) &&
                it.box.width >= minimumWidth &&
                it.box.height >= 20f * density &&
                it.box.left < frame.right && it.box.right > frame.left &&
                it.box.top < frame.bottom && it.box.bottom > frame.top &&
                (it.focused || it.box.bottom >= frame.top + frame.height * .32f)
        }
        if (eligible.isEmpty()) return null
        // Deduplicate the editor's semantic wrapper and native editable child.
        val groups = mutableListOf<MutableList<ComposerCandidate>>()
        for (candidate in eligible.sortedByDescending { it.box.bottom }) {
            val group = groups.firstOrNull { nearSameField(it.first().box, candidate.box, density) }
            if (group != null) group.add(candidate) else groups.add(mutableListOf(candidate))
        }
        val ranked = groups.map { group ->
            group.maxWith(compareBy<ComposerCandidate> { it.setText }
                .thenBy { it.editable }.thenBy { it.focused }
                .thenBy { it.index })
        }.sortedWith(compareByDescending<ComposerCandidate> { it.box.bottom }
            .thenByDescending { it.focused }.thenByDescending { it.setText })

        val top = ranked.first()
        val next = ranked.getOrNull(1)
        // Two competing distinct text areas near the same screen edge are
        // genuinely ambiguous. Do not type into an arbitrary one.
        if (next != null &&
            abs(top.box.bottom - next.box.bottom) <= max(28f * density, frame.height * .035f)) return null
        return top.index
    }
}
