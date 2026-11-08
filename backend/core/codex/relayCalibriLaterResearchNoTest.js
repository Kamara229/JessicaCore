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
    // S62 native-C work is delegated to ChatGPT in GitHub, not Jessica.
    // During a 429 outage this explicit no-test relay must not call Groq.
    const legacyNext=/Следующ(?:ий|ая)\s+шаг\s*[—–:-]\s*\S+/iu.test(body);
    const delegatedGithubNext=/Следующ(?:ий|ая)\s+шаг\s+(?:выполняю|выполняет\s+ChatGPT)\s+через\s+GitHub\s*:\s*\S+/iu.test(body);
    const oldNoTest=/(?:Jessica|Джессика)\s+тесты?\s+не\s+запуска(?:ет|ю|й|ть)\s*[.!]?$/iu.test(body);
    const delegatedNoTest=/(?:Jessica|Джессика)\s+не\s+тестировать\s*[.!]?$/iu.test(body);
    // S63+ status: "Следующий шаг — S63: ChatGPT ... Работа только
    // в GitHub research. Jessica не тестировать." This is a passive
    // research handoff, not a command to execute a Jessica test.
    const numberedResearchNext=/Следующ(?:ий|ая)\s+шаг\s*[—–:-]\s*S\d{1,3}\s*:\s*ChatGPT\s+[^.]{5,200}\./iu.test(body);
    const githubOnly=/Работа\s+только\s+в\s+GitHub\s+research\s*[.!]/iu.test(body);
    const explicitGithubOnlyNoTest=numberedResearchNext && githubOnly && delegatedNoTest;
    // S64 -> S65 passive no-test handoff is phrased as completing the
    // already-started experiment, not "ChatGPT executes through GitHub".
    // Recognize only the constrained, numbered real research completion
    // with both a GitHub PR and a terminal no-Jessica-test statement.
    const numberedGitHubCompletion=/(?:^|\s)S6[45](?=\s|[:—–-])/iu.test(body) &&
        /Следующ(?:ий|ая)\s+шаг\s*[—–:-]\s*завершить\s+обучение\s+S65\s+в\s+GitHub\s+PR\s+#\d+/iu.test(body) &&
        delegatedNoTest;

    if(!(legacyNext || delegatedGithubNext) ||
       !/(?:FP32|TRAIN|Green Tunnel|bpp|QA|навык|обучени|исследован|native\s+C|CB59|скорост|весах|CPU)/iu.test(body) ||
       !(oldNoTest || (delegatedGithubNext && delegatedNoTest) || explicitGithubOnlyNoTest || numberedGitHubCompletion))
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
