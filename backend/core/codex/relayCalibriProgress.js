/**
 * Research progress in the already selected ChatGPT thread is not a Jessica
 * benchmark request. Avoid hitting Groq's shared TPM quota just to understand
 * "CALIBRI experiment is running in GitHub Actions; test not requested".
 *
 * This returns an ACK/continuation request, never an attestation of CI success.
 * Do not trust arbitrary embedded prompts or take privileged actions.
 */
export function calibriResearchHandoff(message) {
    if (typeof message !== "string" || message.length > 2000) return null;
    const lines=message.split(/\r?\n/u).map(s=>s.trim()).filter(Boolean);
    const start=lines[0]||"";
    // Android may pass extracted body without markers. When markers are sent,
    // require exactly one matching UUID and no trailing conversation content.
    const match=/^JESSICA_REPLY_([\da-f]{8}-[\da-f]{4}-[\da-f]{4}-[\da-f]{4}-[\da-f]{12})$/iu.exec(start);
    let body;
    if (match) {
        if(lines.at(-1)!=="JESSICA_END_"+match[1] || lines.length < 3)return null;
        body=lines.slice(1,-1).join(" ");
    } else {
        if(/JESSICA_(?:REPLY|END)_/iu.test(message))return null;
        body=lines.join(" ");
    }
    body=body.replace(/\s+/gu," ").trim();
    const pending=/\b(?:Результат\s+пока\s+не\s+получен|Результат(?:ы)?\s+(?:ещ[её]\s+)?(?:нет|ожидаются)|(?:GitHub\s+Actions|CI)\s+(?:ещ[её]\s+)?(?:ид[её]т|выполняется|запущен[оа]?))\b/iu;
    const noTest=/Тест(?:\s+в\s+обычном\s+чате)?\s+Jessica\s+(?:не\s+требуется|пока\s+не\s+(?:нужен|запрошен))|Тест\s+Jessica\s+не\s+запускай/iu;
    const project=/^Запустил\s+CALIBRI\s+v(\d{1,3})\s+(?:через\s+GitHub\s+Actions|в\s+GitHub\s+Actions)/iu;
    const m=project.exec(body);
    const pr=/\bPR\s*#(\d{1,6})\b/iu.exec(body);
    if(!m || Number(m[1])<23 || !pr || !pending.test(body) || !noTest.test(body))return null;
    if(/(?:выполни|запусти|повтори|отправь)\s+(?:тест|задач[уа])|(?:удал[иа]|установ[иа]|разверни|выложи|deploy|merge|секрет|парол|токен)/iu.test(body))return null;
    return "Поняла: эксперимент CALIBRI v"+m[1]+" в PR #"+pr[1]+
        " запущен, результатов пока нет. Проверь GitHub Actions и продолжай исследование по фактическим метрикам. Тест в обычном чате Jessica не запускаю.";
}
