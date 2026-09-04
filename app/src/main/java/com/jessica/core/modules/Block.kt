package com.jessica.core.modules

data class Block(

    val id: String,

    val name: String,

    val version: String,

    val author: String,

    val type: String,

    val status: String,

    val description: String,

    val capabilities: List<String>

)
