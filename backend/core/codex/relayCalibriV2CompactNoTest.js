/** Unambiguous compact CALIBRI 2.0 research handoff is NOT a Jessica test.
 *
 * Reproduces the real S24 incident: "S24: обучение ... запущено в GitHub.
 * Следующий шаг — проверить ... Jessica тесты не запускает."
 * The word "проверить" describes what ChatGPT checks in GitHub; it must
 * NEVER be extracted as an ordinary-chat task for Jessica.
 *
 * Fail closed for malformed JESSICA envelopes, external actions, embedded
 * explicit test requests or extra untrusted trailing text.
 */
export function compactCalibri2NoJessicaTest(message) {
    if (typeof message !== "string" || !message.trim() || message.length > 1300)
        return null;
    const lines=message.trim().split(/\r?\n/u).map(s=>s.trim()).filter(Boolean);
    const envelope=/^JESSICA_REPLY_([\da-f]{8}-[\da-f]{4}-[\da-f]{4}-[\da-f]{4}-[\da-f]{12})$/iu.exec(lines[0]||"");
    let body;
    if(envelope){
        if(lines.length<3 || lines.at(-1)!=="JESSICA_END_"+envelope[1])return null;
        body=lines.slice(1,-1).join(" ");
    }else{
        if(/JESSICA_(?:REPLY|END)_/iu.test(message))return null;
        body=lines.join(" ");
    }
    body=body.replace(/\s+/gu," ").trim();
    if(body.length>600)return null;
    // JS \b is ASCII-only; Russian word boundaries must be Unicode-safe.
    // CALIBRI research may report either "S24: обучение ... запущено"
    // or "S24 ещё обучается в GitHub"; both are no-test status notes.
    // JS \\b word boundary does not match Russian letters reliably.
    const numbered=/^S(\\d{1,3})(?:(?:\\s*:\\s*(?:обучение|эксперимент|исследование)(?=\\s|[.,:;—–-]))|(?:\\s+ещ[её]\\s+обучается\\s+в\\s+GitHub(?=[.?!\\s])))/iu.exec(body);
    if(!numbered || Number(numbered[1])<1)return null;
    if(!/\bGitHub\b/iu.test(body) ||
       !/Следующ(?:ий|ая)\s+шаг\s*[—–:-]\s*(?:после\s+завершения\s+)?(?:проверить|сверить|изучить|дождаться)(?=\s)/iu.test(body) ||
       !/(?:Jessica|Джессика)\s+тесты?\s+не\s+запуска(?:ет|етcя|й|ть)\s*[.!]?$/iu.test(body))
       return null;
    if(/(?:создай|запусти|выполни|проведи|повтори|открой)\s+(?:новый\s+|следующий\s+|этот\s+|один\s+)?(?:тест|тестовый\s+чат|задач[уа])/iu.test(body) ||
       /(?:точная\s+(?:задача|формулировка|инструкция)|в\s+обычном\s+чате|через\s+обычный\s+чат|парол|секрет|API[- ]?ключ|токен\s+(?:доступа|авторизац)|удали|установи|deploy|merge|https?:\/\/|[\x60«»])/iu.test(body) ||
       /(?:план|исследование)\s+(?:полностью\s+)?заверш[её]н/iu.test(body))
       return null;
    return {
      action:"reply",
      task:"",
      continueThread:false,
      reply:"Поняла: CALIBRI 2.0, S"+numbered[1]+
        ". Этот шаг выполняет ChatGPT через GitHub. "+
        "Новый тест в обычном чате Jessica не запрашивался и не запускался. "+
        "Передаю новые указания владельца; результатов не придумываю."
    };
}
