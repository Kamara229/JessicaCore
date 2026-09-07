package com.jessica.core.modules.chat


data class ChatMessage(

    val text: String,

    val isUser: Boolean,

    val timestamp: Long = System.currentTimeMillis()

)
