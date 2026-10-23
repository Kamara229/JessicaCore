package com.jessica.core.modules.codex.chat

/**
 * Safe verification for a draft that ChatGPT may re-render asynchronously.
 * Only known non-semantic Android text transformations are tolerated.
 * No automatic resend, no replacement of an unrelated owner's draft.
 */
internal object CodexDraftVerificationPolicy {
    private fun normalize(text: String): String =
        text.replace("\r\n","\n").replace('\u00a0',' ')
            .replace("\u200b","").replace("\u2060","").trimEnd('\n','\r')

    fun matchesExpected(actual: String, expected: String): Boolean =
        expected.isNotBlank() && normalize(actual) == normalize(expected)
}
