package com.jessica.core.modules.chat

/** Full text page; offsets count UTF-16 code units, matching the backend. */
data class DocumentPage(
    val name: String,
    val text: String,
    val truncated: Boolean,
    val extractedChars: Int,
    val offset: Int,
    val nextOffset: Int?,
    val hasMore: Boolean,
    val sourceSha256: String
)
