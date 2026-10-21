/**
 * Select only the latest usable source-code exchange for a code-edit followup.
 * Earlier unrelated code examples must not poison the edit target or increase
 * Planner token use. No content is executed or promoted to reusable Experience.
 */
export function recentCodeEditContext(task,input) {
    const current=String(task||"").trim();
    if(!/^(?:(?:а\s+)?теперь\s+)?(?:добавь|исправь|поправь|перепиши|измени|доработай|fix|update|modify|refactor)/iu.test(current)) return null;
    if(!/(?:функци|код|скрипт|python|kotlin|javascript|typescript|java|rust|c\+\+|program|function|script|source|tests?|алгоритм)/iu.test(current)) return null;
    if(!Array.isArray(input)) return null;
    const history=input.slice(-12).filter(item =>
        item && ["user","assistant"].includes(item.role) &&
        typeof item.content === "string" && item.content.trim());
    const index=history.findLastIndex(item=>item.role==="assistant" &&
        item.content.trim().length>15 &&
        !/^(?:ошибка сервера|ошибка подключения|server error|http 50[234]|could not|не удалось завершить)/iu.test(item.content.trim()));
    if(index<0) return null;
    const response=history[index].content.trim();
    const query=history.slice(0,index).reverse().find(item=>item.role==="user")?.content.trim()||"";
    const summary=response.length>2200
        ? response.slice(0,1600)+"\n[...предыдущий ответ сокращён...]\n"+response.slice(-480)
        : response;
    return [
        ...(query?[{role:"user",content:query.slice(0,450)}]:[]),
        {role:"assistant",content:summary}
    ];
}
