import {explicitSameTestChat} from "./relayThreadPolicy.js";

/**
 * Deterministically route a SINGLE explicitly requested Jessica test.
 * Do not ask the owner for the result or replace real execution with a
 * model-generated expected answer. The returned task must be a verbatim
 * substring of the original ChatGPT message.
 */
export function explicitJessicaTest(message) {
    const raw=typeof message === "string" ? message : "";
    if (!raw || raw.length>50000) return null;
    const lines=raw.split(/\r?\n/);
    const commands=lines.filter(x => !/^\s*(?:не|НЕ)\s+/u.test(x.trim().replace(/^\*+|\*+$/g,"")));
    const explicitCommand=commands.some(x =>
        /(?:создай|создать|открой|открыть)\s+нов(?:ый|ом)\s+тестов(?:ый|ом)\s+чат\s+(?:Jessica|Джессик)/iu.test(x)
        || /(?:выполни|выполнить|запусти|запустить)\s+тест\s+(?:через|в)\s+(?:обычн\w*\s+)?чат(?:е)?\s+(?:Jessica|Джессик)/iu.test(x)
        || /(?:продолжи|продолжить|используй)\s+в\s+том\s+же\s+(?:тестов\w*\s+)?чат(?:е)?\s+(?:Jessica|Джессик)/iu.test(x)
    );
    if (!explicitCommand) return null;
    const header=/^\s*(?:\*\*)?(?:Точная\s+формулировка\s+(?:задачи|теста)|Точная\s+инструкция|Точная\s+задача)(?:\*\*)?\s*:\s*(?:\*\*)?\s*$/iu;
    const headings=lines.flatMap((line,index)=>header.test(line)?[index]:[]);
    if (headings.length!==1) return null;
    let from=headings[0]+1;
    while (from<lines.length && !lines[from].trim()) from++;
    if (from>=lines.length) return null;
    let to=from;
    while(to<lines.length && lines[to].trim() && !/^\s*(?:\*\*)?(?:Ожидаемый\s+результат|Верни\s+фактический\s+ответ|После\s+получения\s+результата)/iu.test(lines[to]))to++;
    const task=lines.slice(from,to).join("\n").trim();
    if(!task || task.length>12000 || !raw.includes(task))return null;
    // Pure instruction, not a label/example or other output directive.
    if (/^(?:\*\*|#{1,6}\s|>|JESSICA_(?:REPLY|END)_)/u.test(task)) return null;
    return {action:"test",reply:"",task,continueThread:explicitSameTestChat(raw)};
}
