package com.jessica.core.modules.codex.relay

/** Natural-language conversation; unique envelopes distinguish complete new replies from old text. */
object RelayProtocol {
    const val GREETING = "Джессика готова к работе в чате."
    val introduction = "$GREETING\nПродолжаем уже составленный тобой план улучшения backend Jessica Core. " +
        "Ты работаешь со своими подключениями GitHub, Render и Supabase. Я не редактирую репозиторий и не управляю развёртыванием. " +
        "Я читаю твои сообщения, выполняю предложенные текстовые тесты через свой обычный чат и возвращаю точный ответ. " +
        "Давай тесты по одному с полной формулировкой; для продолжения предыдущего теста явно укажи тот же чат. " +
        "Не проси устанавливать APK, менять настройки телефона, вводить секреты или подтверждать действия владельца. " +
        "Ответ для меня делай коротким: итог текущего шага и один тест либо следующий шаг; полный код сюда не копируй. " +
        "Если тест пока не нужен, продолжай план; когда план завершён, сообщи об этом."

    fun wrap(text: String, id: String): String = text +
        "\n\nДля надёжного чтения завершённого ответа: начни ответ отдельной строкой JESSICA_REPLY_$id; " +
        "между маркерами пиши обычным текстом, не JSON; закончи отдельной строкой JESSICA_END_$id. " +
        "Метки относятся только к ответу на это сообщение."

    fun reply(screen: String, id: String): String? {
        if (screen.length > 120000 || !Regex("[a-f0-9-]{36}").matches(id)) return null
        val lines = screen.replace("\r\n", "\n").lines()
        val first = lines.indexOfFirst { it.trim() == "JESSICA_REPLY_$id" }
        if (first < 0) return null
        val end = (first + 1 until lines.size).firstOrNull { lines[it].trim() == "JESSICA_END_$id" } ?: return null
        return lines.subList(first + 1, end).joinToString("\n").trim().takeIf { it.isNotBlank() && it.length <= 50000 }
    }
}
