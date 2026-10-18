/*
 * Compact comparison claim verifier. The Qwen Groq tier has a strict
 * per-minute OUTPUT budget; the older 1800-token response allocation
 * exceeded a verified 1000 OTPM account limit and could never succeed.
 *
 * Keep citations verifiable by the local original-source matcher.
 */
export function buildComparisonClaimInstructions() {
    return [
        "Ты проверяешь сравнительную таблицу по двум документам.",
        "Не отвечай пользователю. Не пользуйся внешними знаниями.",
        "Верни только JSON object вида:",
        '{"claims":[{"claim":"краткий факт","supported":true,"evidence":"дословная короткая цитата","sourceUrl":"URL","reason":""}]}',
        "Проверь каждую существенную строку таблицы: обе сравниваемые стороны.",
        "Обычно нужно 2 факта на строку и 2 независимых URL.",
        "claim: одно короткое проверяемое утверждение (до 70 символов).",
        "evidence: 5–16 соседних слов из ТОГО ЖЕ источника (до 110 символов).",
        "Не соединяй цитаты из разных мест через многоточие.",
        "Цитаты сохраняют исходные слова и порядок. Markdown-разметку можно убрать.",
        "sourceUrl должен точно соответствовать документу, который содержит цитату.",
        "Если факт отсутствует или невозможно проверить: supported=false, evidence=''.",
        "Не доказывай отсутствие условия отсутствием одной фразы.",
        "Не пропускай утверждения о различиях или исключениях.",
        "reason всегда пустая строка. JSON должен быть кратким.",
        "Не добавляй объяснений вне JSON."
    ].join("\n");
}
