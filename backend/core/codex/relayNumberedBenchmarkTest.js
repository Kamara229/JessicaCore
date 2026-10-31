/**
 * A single numbered benchmark task is a direct Codex instruction.
 * Extract only its literal one-line task; never ask the quota-bound LLM.
 */
export function numberedBenchmarkTest(message) {
    const raw = typeof message === "string" ? message : "";
    if (!raw || raw.length > 12000) return null;
    const body = raw.split(/\r?\n/).filter(line =>
        !/^\s*JESSICA_(?:REPLY|END)_[a-f\d-]{36}\s*$/iu.test(line)
    ).join("\n");
    if (/(?:не\s+(?:отправляй|запускай|выполняй|повторяй)|например|(?:^|\s)пример\s*:|образец\s*:)/iu.test(body)) return null;
    // A bounded '(повтор ...)' note does not change the verbatim task.
    const matches = [
        ...body.matchAll(/^\s*Тест\s+([1-9]\d{0,2})\s*\/\s*([1-9]\d{0,2})\s*(?:\(\s*повтор[^()\r\n]{0,80}\))?\s*:\s*(.+?)\s*$/gimu),
        // A previous result and the next task can share one text line.
        ...body.matchAll(/(?:^|[.!?]\s+)Тест\s+([1-9]\d{0,2})\s*\/\s*([1-9]\d{0,2})\.\s+(Задача\s+[^\r\n]+?)\s*$/gimu)
    ];
    if (matches.length !== 1) return null;
    const [, sequence, total, rawTask] = matches[0];
    if (Number(sequence) > Number(total) || Number(total) > 200) return null;
    const task = rawTask.trim();
    if (task.length < 20 || task.length > 1800) return null;
    if (/(?:^|[.!?]\s+)Тест\s+[1-9]\d{0,2}\s*\/\s*[1-9]\d{0,2}\s*(?::|\.\s+Задача\s+)/iu.test(task)) return null;
    return {action:"test", reply:"", task, continueThread:false};
}
