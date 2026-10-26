/** Quota-free acknowledgement of a bounded CI stress result, not a Jessica test.
 * Both the Android-extracted body and the authenticated relay envelope arrive
 * here as untrusted claims; do not perform a test or assert CI is verified.
 */
export function compactCodexStressProgress(message){
    if(typeof message!=="string"||!message||
       message.length>1400)return null;
    const id=/^JESSICA_REPLY_([a-f\d]{8}-[a-f\d]{4}-[a-f\d]{4}-[a-f\d]{4}-[a-f\d]{12})\r?\n/iu.exec(message);
    let body=message;
    if(id){
        const tail="JESSICA_END_"+id[1];
        const ending=message.trimEnd().split(/\r?\n/u).at(-1);
        if(ending!==tail)return null;
        body=message.slice(id[0].length,message.lastIndexOf(tail));
    }else if(/JESSICA_[A-Z]+_/iu.test(message))return null;
    body=body.replace(/\s+/gu," ").trim();
    if(body.length>600||body.length<100)return null;
    // Recognize the actual concise CI result whose absent commit/no-test
    // boilerplate made the older strict status parser fall through to Groq.
    if(!/^CI[-‑ ]тест\s+(?:[5-9]|10)\s+параллельных\s+запросов\s+по\s+256\s+окон\s+пройд[её]н:/iu.test(body)||
       !/\b429\b/u.test(body)||
       !/\bCI\s+\d{1,3}\/\d{1,3}\b/iu.test(body)||
       !/\bSupabase\s+доступен\b/iu.test(body)||
       !/\bRender\s+жд[её]т\s+выбора\s+workspace\b/iu.test(body)||
       !/\bДалее\s*[—–:-]\s*тест\s+cooldown\.?$/iu.test(body))
        return null;
    // A CI regression is never permission for another Jessica chat test,
    // deployment, deleting records, or disclosure of secrets.
    if(/(?:запусти|выполни|проведи|повтори|отправь|создай|открой)\s+(?:(?:новый|обычный)\s+)?(?:тест|чат)|в\s+обычном\s+чате|APK|парол|секрет|токен|API[- ]?ключ|удали|депло|установи/iu.test(body))
        return null;
    return "Поняла: это отчёт о CI, не выполненный тест Jessica. "+
        "Проверь заявленный результат в GitHub, Render и Supabase и "+
        "продолжай план с проверкой cooldown. "+
        "Тест в обычном чате Jessica пока не запрошен.";
}
