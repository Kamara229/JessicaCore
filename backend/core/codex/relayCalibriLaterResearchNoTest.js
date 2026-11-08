/**
 * CALIBRI S28+ progress (including S30 FP32 TRAIN skill regression) is a
 * research instruction for main ChatGPT/GitHub, not a Jessica ordinary-chat
 * test. This is the fail-closed provider-free path for the 2026-11-08 outage.
 *
 * No external tools, GitHub calls, training or QA are performed here.
 * Never trust unmatched markers, ambiguous follow-up, secrets or test requests.
 */
export function calibriLaterResearchNoJessicaTest(message) {
    if (typeof message!=="string" || !message.trim() || message.length>1500) return null;
    const lines=message.trim().split(/\r?\n/u).map(x=>x.trim()).filter(Boolean);
    const match=/^JESSICA_REPLY_([\da-f]{8}-[\da-f]{4}-[\da-f]{4}-[\da-f]{4}-[\da-f]{12})$/iu.exec(lines[0]||"");
    let body;
    if(match){
        if(lines.length<3 || lines.at(-1)!=="JESSICA_END_"+match[1])return null;
        body=lines.slice(1,-1).join(" ");
    }else{
        if(/JESSICA_(?:REPLY|END)_/iu.test(message))return null;
        body=lines.join(" ");
    }
    body=body.replace(/\s+/gu," ").trim();
    if(!body || body.length>800)return null;
    const version=/^(?:CALIBRI\s+)?S(2[89]|[3-9]\d|1\d\d)(?=\s|[:.,;—–-])/iu.exec(body);
    if(!version)return null;
    // S38 incident: owner gave "Следующий шаг ChatGPT — S39" and
    // "Jessica не тестировать, LIVE не менять". Older parser required the
    // phrase "Jessica тесты не запускает" at the very end; an explicit
    // protected GitHub-only research step fell into quota-limited Groq.
    const legacyNoTest=/(?:Jessica|Джессика)\s+тесты?\s+не\s+запуска(?:ет|ю|й|ть)\s*[.!]?$/iu.test(body);
    const researchOnlyNoTest=/(?:Jessica|Джессика)\s+не\s+тестировать\s*,\s*LIVE\s+не\s+меня(?:ть|й)(?=\s|[.!?]|$)/iu.test(body);
    // S40 incident: a short research-only instruction starts "Шаг S41:"
    // instead of "Следующий шаг ChatGPT —". The old strict phrase guard
    // missed it and needlessly called the rate-limited interpreter.
    const hasResearchStep =
        /Следующ(?:ий|ая)\\s+шаг(?:\\s+ChatGPT)?\\s*[—–:-]\\s*\\S+/iu.test(body) ||
        /(?:^|[.!?]\\s+)Шаг\\s+S(?:2[89]|[3-9]\\d|1\\d\\d)\\s*:\\s*(?:тренировать|обучать|диагностир\\w*|сравни\\w*|исследов\\w*|проверить|проанализиров\\w*|изучить|улучш\\w*)/iu.test(body);
    if(!hasResearchStep ||
       !/(?:FP32|TRAIN|Green Tunnel|bpp|QA|навык|обучени|обучать|исследован|сжатие|кодовые\s+книги)/iu.test(body) ||
       !(legacyNoTest || researchOnlyNoTest))
       return null;
    // A status note must not accidentally suppress an actual Jessica chat test,
    // credential request, external action or an explicitly finished plan.
    if(/(?:создай|открой|выполни|запусти|проведи|повтори|отправь|протестируй)\s+(?:(?:один|этот|новый|следующий)\s+)?(?:тест(?:ы)?|тестовый\s+чат|задач[уа])/iu.test(body) ||
       /(?:в\s+обычном\s+чате|через\s+обычный\s+чат|Точная\s+(?:формулировка|задача|инструкция)|парол|секрет|API[- ]?ключ|токен\s+(?:доступа|авторизац\w*|GitHub|API)|удали|установи|deploy|merge|https?:\/\/|[\x60«»])/iu.test(body) ||
       /(?:план|исследование)\s+(?:полностью\s+)?заверш[её]н/iu.test(body))
       return null;
    return {
      action:"reply",task:"",continueThread:false,
      reply:"Поняла: CALIBRI S"+version[1]+". Следующий исследовательский шаг "+
            "выполняет ChatGPT через GitHub, не Jessica. "+
            "Тест в обычном чате не запрашивался и не запускался. "+
            "Продолжайте работу с фактическими результатами, без домыслов."
    };
}
