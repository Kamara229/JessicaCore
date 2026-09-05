package com.jessica.core.modules


data class AIResult(
    val success: Boolean,
    val text: String
)


interface AIEngine {

    suspend fun solve(
        task: String
    ): AIResult

}
