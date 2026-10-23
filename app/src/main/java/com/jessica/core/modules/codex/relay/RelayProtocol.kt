package com.jessica.core.modules.codex.relay

/** Natural-language conversation; unique envelopes distinguish complete new replies from old text. */
object RelayProtocol {
    const val GREETING = "Джессика готова к работе в чате."
    // Keep the first Android message short: on some ChatGPT builds a long
    // multiline composer changes its accessibility geometry before sending.
    val introduction = "$GREETING\nПродолжаем план backend Jessica Core. " +
        "GitHub, Render и Supabase меняешь ты; я не редактирую репозиторий. " +
        "Я выполняю тесты через свой обычный чат и возвращаю дословные результаты. " +
        "Давай один тест либо следующий шаг, без APK, настроек и секретов. " +
        "Когда план завершён — сообщи."

    fun wrap(text: String, id: String): String = text +
        "\n\nНачни ответ отдельной строкой JESSICA_REPLY_$id. " +
        "Между метками не более 240 символов: только один тест или следующий шаг, без отчётов. " +
        "Закончи отдельной строкой JESSICA_END_$id."

    fun reply(screen: String, id: String): String? =
        ChatReplyExtraction.read(screen, id)
}
