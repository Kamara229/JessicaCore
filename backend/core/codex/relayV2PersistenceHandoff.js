/** Narrow, no-model handoff for a rolled-back Skill v2 RPC probe.
 * Never treats this as a completed CI run or an ordinary Jessica test.
 */
export function v2PersistenceHandoff(message) {
    if(typeof message!=="string"||!message||message.length>1800)return null;
    const marker=/^JESSICA_REPLY_([a-f\d]{8}-[a-f\d]{4}-[a-f\d]{4}-[a-f\d]{4}-[a-f\d]{12})\s*\r?\n/iu.exec(message);
    let body=message;
    if(marker) {
        const end="JESSICA_END_"+marker[1];
        if(message.trimEnd().split(/\r?\n/u).at(-1)?.trim()!==end)return null;
        body=message.slice(marker[0].length,message.lastIndexOf(end));
    } else if(/JESSICA_(?:REPLY|END)_/iu.test(message))return null;
    body=body.replace(/\s+/gu," ").trim();
    if(!/^RPC\s*v4\s*:/iu.test(body)||
       !/сохранение\s+Skill\s+с\s+манифестом\s+v2\s+подтверждено/iu.test(body)||
       !/транзакция\s+откатана\s+без\s+изменений\s+БД/iu.test(body)||
       !/CI\s+пока\s+не\s+подтвержд[её]н/iu.test(body)||
       !/следующий\s+шаг\s*[—–:-]\s*проверить\s+восстановление\s+v2\s+и\s+отклонение\s+поврежд[её]нных\s+модулей\.?$/iu.test(body))
        return null;
    if(/(?:запусти|выполни|повтори|проведи|отправь)\s+(?:новый\s+)?тест|\bAPK\b|секрет|парол|ключ\s+доступа/iu.test(body))
        return null;
    return "Поняла: RPC v4 проверен откатываемой транзакцией, CI не подтверждён. "+
      "Проверь восстановление v2 и отказ на повреждённых модулях через GitHub, Render и Supabase; "+
      "продолжай план. Тест в обычном чате Jessica не запрошен.";
}
