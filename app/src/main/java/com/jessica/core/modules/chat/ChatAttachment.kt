package com.jessica.core.modules.chat

data class ChatAttachment(
    val id: String,
    val name: String,
    val mime: String,
    val sizeBytes: Long
)

data class DocumentContext(
    val name: String,
    val text: String,
    val truncated: Boolean,
    val pages: List<DocumentPage> = emptyList()
)
