/**
 * One owner-requested Jessica task may arrive from Android as a verified
 * bounded ChatGPT reply BODY (the outer JESSICA_REPLY/END lines are stripped).
 *
 * Prior versions only understood "Отправь в обычном чате Jessica: «...»".
 * The common exact instruction "Тест: ... Рассчитай ..." fell through to a
 * quota-bound AI interpreter and could yield interpret_failed with NO test.
 *
 * This is intentionally narrow: no test inference from plans, examples,
 * multiple commands, malformed envelopes, failures, quoted status or prose.
 * A test is never executed by this module; it only identifies its exact text.
 */
export function onePlainJessicaTest(message) {
    const source=typeof message==="string"?message:"";
    if(!source || source.length>1400)return null;
    const trimmed=source.trim();
    const marker=/^JESSICA_REPLY_([a-f\d]{8}-[a-f\d]{4}-[a-f\d]{4}-[a-f\d]{4}-[a-f\d]{4}-[a-f\d]{12})\s*\r?\n/iu.exec(trimmed);
    let body;
    if(marker){
        const suffix="JESSICA_END_"+marker[1];
        if(trimmed.split(/\r?\n/u).at(-1)?.trim()!==suffix)return null;
        body=trimmed.slice(marker[0].length,trimmed.lastIndexOf(suffix)).trim();
    } else {
        if(/JESSICA_(?:REPLY|END)_/iu.test(trimmed))return null;
        body=trimmed;
    }
    if(body.length>240 || body.includes("\n"))return null;
    if(/(?:не\s+(?:отправляй|повторяй|запускай|выполняй|тестируй)|например|пример\s*:|гипотез|тест\s+не\s+(?:прош[её]л|выполнялся|запускался)|результат\s+теста)/iu.test(body))
        return null;
    const match=/^Тест\s*:\s*(.{30,220})$/iu.exec(body);
    if(!match)return null;
    const task=match[1].trim();
    if(!/(?:рассчитай|вычисли|проверь|определи|найди|реши|сравни|покажи)(?=$|[^\p{L}\p{N}])/iu.test(task))
        return null;
    if(/(?:^|[.!?]\s+)Тест\s*:/iu.test(task))return null;
    // This is a single verbatim task. No fabrication, no source changes.
    if(!source.includes(task))return null;
    return {action:"test",reply:"",task,continueThread:false};
}
