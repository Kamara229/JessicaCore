/**
 * S26 still-training/S27 next-audit is a future GitHub research status,
 * NEVER an ordinary-chat Jessica test. Handle the exact 2026-11-08 incident
 * without Groq/LLM interpretation. Do not claim research already finished.
 *
 * Fail closed for explicit Jessica chat tests, secret actions, malformed
 * envelopes and appended instructions. All model selection, CPU and full-EOS
 * audits remain exclusively with the main ChatGPT GitHub research workflow.
 */
export function pendingCalibrI26AuditHandoff(message) {
    if (typeof message !== "string" || !message.trim() || message.length > 1400) return null;
    const lines=message.trim().split(/\r?\n/u).map(x=>x.trim()).filter(Boolean);
    const open=/^JESSICA_REPLY_([\da-f]{8}-[\da-f]{4}-[\da-f]{4}-[\da-f]{4}-[\da-f]{12})$/iu.exec(lines[0]||"");
    let body;
    if(open){
        if(lines.length<3 || lines.at(-1)!=="JESSICA_END_"+open[1])return null;
        body=lines.slice(1,-1).join(" ");
    }else{
        if(/JESSICA_(?:REPLY|END)_/iu.test(message))return null;
        body=lines.join(" ");
    }
    body=body.replace(/\s+/gu," ").trim();
    if(body.length>820)return null;
    // Explicit research CI status, not an unqualified "run test".
    if(!/\bS26\b/iu.test(body) || !/\bS27\b/iu.test(body) ||
       !/\bGitHub\b/iu.test(body) || !/\bCI\b/iu.test(body) ||
       !/(?:ещ[её]\s+(?:обучается|выполняет\s+обучение)|продолжа(?:ет|ется)\s+обучение)/iu.test(body) ||
       !/Следующий\s+шаг\s*[—–:\-]\s*после\s+завершения\s+CI\s+запустить\s+аудит\s+S27\s*:\s*проверить\s+96\s+ответов/iu.test(body) ||
       !/сохранени[ея]\s+навыков/iu.test(body) ||
       !/\bbpp\b/iu.test(body) || !/\bCPU\b/iu.test(body) ||
       !/(?:Jessica|Джессика)\s+тесты\s+не\s+запускает\s*[.!]?$/iu.test(body))
       return null;
    if (/(?:создай|открой|запусти|выполни|проведи|повтори|отправь)\s+(?:(?:новый|следующий|этот|один)\s+)?(?:тест|тестовый\s+чат|задач[уа])/iu.test(body) ||
        /(?:в\s+обычном\s+чате|через\s+обычный\s+чат|Точная\s+(?:задача|формулировка|инструкция)|парол|секрет|API[- ]?ключ|токен\s+(?:доступа|авторизац)|удали|установи|deploy|merge|https?:\/\/|[\x60«»])/iu.test(body))
        return null;
    return {
        action:"reply",task:"",continueThread:false,
        reply:"CALIBRI S26 обучается в GitHub. Будущий аудит S27 (96 ответов, навыки, bpp, CPU) выполняет ChatGPT после завершения CI. Jessica тесты не запускает и результаты не придумывает."
    };
}
