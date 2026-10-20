package com.jessica.core.modules.chat

/** A deleted chat stays recoverable for exactly one hour from confirmation. */
object ChatTrashPolicy {
    const val RETENTION_MS: Long = 60L * 60L * 1000L

    fun expiresAt(deletedAt: Long?): Long? = deletedAt?.plus(RETENTION_MS)

    fun isExpired(deletedAt: Long?, now: Long = System.currentTimeMillis()): Boolean =
        deletedAt != null && now >= deletedAt + RETENTION_MS

    fun canRestore(deletedAt: Long?, now: Long = System.currentTimeMillis()): Boolean =
        deletedAt != null && !isExpired(deletedAt, now)

    fun minutesLeft(deletedAt: Long?, now: Long = System.currentTimeMillis()): Long {
        val expires = expiresAt(deletedAt) ?: return 0
        val remaining = (expires - now).coerceAtLeast(0)
        return (remaining + 59_999L) / 60_000L
    }
}
