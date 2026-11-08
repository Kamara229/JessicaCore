/**
 * CALIBRI GitHub-only bounded progress does not need Groq interpretation.
 *
 * This is a deliberately narrow, non-executing acknowledgment for status
 * messages delegated to ChatGPT, not an ordinary Jessica chat test.
 * Do not report unobserved success or launch any work from a status.
 */
export function quotaSafeCalibrIResearchProgress(message) {
    if(typeof message!=="string" || message.length>850)return null;
    const lines=message.trim().split(/\r?\n/u).map(x=>x.trim()).filter(Boolean);
    if(lines.length!==3)return null;
    const marker=/^JESSICA_REPLY_([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$/iu.exec(lines[0]);
    if(!marker || lines[2]!=="JESSICA_END_"+marker[1])return null;
    const body=lines[1].replace(/\s+/gu," ").trim();
    if(body.length<75 || body.length>380)return null;
    const stage=/(?:^|[^A-Za-z0-9])S(\d{2,3})(?=[^A-Za-z0-9]|$)/iu.exec(body)?.[1];
    if(!stage || Number(stage)<28)return null;
    // GitHub-only research may be cited by a PR number OR an immutable CI run ID.
    // This check only acknowledges a status; it cannot run that CI itself.
    if(!/\bGitHub\b/iu.test(body) ||
       !/(?:\bPR\s*#\d{1,5}\b|\bCI\s*#\d{5,15}\b)/iu.test(body))return null;
    // No test can be inferred from a passive research status.
    if(!(/(?:Jessica|Джессика)\s+не\s+тестировать\s*[.!]?$/iu.test(body) ||
         /Тест\s+(?:Jessica|Джессика)\s+не\s+запускался\s*[.!]?$/iu.test(body)))return null;
    if(!/(?:следующ(?:ий|ая)\s+шаг|продолжаю|запущен|проверка|исследование|обучение)/iu.test(body))return null;
    if(/(?:^|\s)(?:создай|открой|запусти|выполни|проведи|повтори|отправь|измени|удали|разверни|деплой|merge|deploy)\s+(?:[^\s]+\s+){0,3}(?:тест|Jessica|Джессик|чат|Render|Supabase|LIVE|APK)/iu.test(body))return null;
    if(/(?:https?:\/\/|парол|секрет|API[- ]?ключ|токен\s+доступа|[«»\u0060]|JESSICA_(?:REPLY|END)_)/iu.test(body))return null;
    return {action:"reply",task:"",continueThread:false,
      reply:"Поняла: CALIBRI S"+stage+". Статус GitHub принят без запроса к Groq. "+
        "Тест в обычном чате Jessica не запрашивался и не запускался. "+
        "Исследование и проверки проводит ChatGPT; передаю следующие сообщения владельца."};
}
