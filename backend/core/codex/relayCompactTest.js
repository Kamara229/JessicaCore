/**
 * Extract one explicitly requested compact Jessica test without LLM calls.
 * The quoted task is retained verbatim; never infer a task from examples.
 */
export function compactJessicaTest(message) {
    const raw = typeof message === "string" ? message : "";
    if (!raw || raw.length > 12000) return null;
    const body = raw.split(/\r?\n/).filter(line =>
        !/^\s*JESSICA_(?:REPLY|END)_[a-f\d-]{36}\s*$/iu.test(line)
    ).join("\n");
    if (/(?:не\s+отправляй|не\s+выполняй|не\s+запускай)/iu.test(body)) return null;
    // Accept both natural word orders and a line break after the colon.
    // The quoted task stays verbatim; no inferred test or provider call.
    const pattern = /(?:^|[^\p{L}\p{N}])(?:В\s+обычном\s+чате(?:\s+(?:Jessica|Джессик[аи]))?\s+отправь|Отправь\s+в\s+обычный\s+чат\s+(?:Jessica|Джессик[аи])|Тест\s+в\s+обычном\s+чате|Следующий\s+тест\s+в\s+обычном\s+чате\s+(?:Jessica|Джессик[аи]))\s*:\s*«([^»\n]{8,2500})»/giu;
    const matches = [...body.matchAll(pattern)];
    if (matches.length !== 1) return null;
    const task = matches[0][1].trim();
    if (!task || !raw.includes(task)) return null;
    return {action:"test",reply:"",task,continueThread:false};
}
