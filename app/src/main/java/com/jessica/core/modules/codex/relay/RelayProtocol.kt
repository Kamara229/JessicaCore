package com.jessica.core.modules.codex.relay

/** Natural-language conversation; unique envelopes distinguish complete new replies from old text. */
object RelayProtocol {
    const val GREETING = "Джессика готова к работе в чате."
    // Keep the first Android message short: on some ChatGPT builds a long
    // multiline composer changes its accessibility geometry before sending.
    val introduction = "$GREETING\nПродолжаем выбранный план чата: backend Jessica Core или исследование CALIBRI. " +
        "GitHub, Render и Supabase меняешь ты; я не редактирую репозиторий. " +
        "Я выполняю тесты через свой обычный чат и передаю ответы дословно. " +
        "Один следующий шаг без APK, настроек и секретов. Когда план завершён — сообщи."

    /** The owner's already-selected ChatGPT thread decides the dialogue goal. */
    val researchIntroduction = "$GREETING\nПродолжаем исследование CALIBRI: новые многомерные нейровеса, " +
        "их физическое сжатие, качество и скорость. GitHub и эксперименты ведёшь ты " +
        "в отдельной research-ветке; рабочую Джессику, Render, Supabase и APK не меняем. " +
        "Я отвечаю вместо владельца в этом чате, прошу продолжать следующий этап, " +
        "при явном запросе выполняю один текстовый тест через обычный чат Jessica. " +
        "Отличай измерение от гипотезы, негативные результаты не скрывай."

    fun introductionFor(selectedChatText: String): String =
        if (Regex("CALIBRI|нейровес|нейро-вес|сжатие весов", RegexOption.IGNORE_CASE)
                .containsMatchIn(selectedChatText)) researchIntroduction else introduction

    fun wrap(text: String, id: String): String = text +
        "\n\nНачни отдельной строкой JESSICA_REPLY_$id. " +
        "Между метками не более 240 символов: один шаг или тест, без отчётов. " +
        "Последняя строка: JESSICA_END_$id."

    fun reply(screen: String, id: String): String? =
        ChatReplyExtraction.read(screen, id)
}
