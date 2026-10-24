/*
 * Evidence safety for free-form answers about the MIT License.
 * The absence of an express patent grant does not itself establish
 * the absence of any patent rights (including potentially implied rights).
 * This is a fail-closed presentation check, never positive verification.
 */

function mitPatentQuestion(task) {
    const question=String(task||"");
    return /\bmit(?:\s+license)?\b/iu.test(question) &&
        /(?:патент|patent)/iu.test(question);
}

export function checkMitPatentOverclaim(task,answer) {
    if(!mitPatentQuestion(task))return null;
    const words=String(answer||"").trim().match(/[\\p{L}\\p{N}]+/gu)||[];
    // Explicit demand for an explanation cannot pass as a bare yes/no.
    if(/(?:обосну|объясн|поясни|explain|justify)/iu.test(String(task||""))
       && words.length<32) {
        return {valid:false,shouldRetry:true,
            reason:"Запрошено обоснование по тексту MIT License, но ответ слишком краток и не раскрывает основание вывода."};
    }
    const paragraphs=String(answer||"").split(/\n\s*\n/);
    for(const paragraph of paragraphs) {
        const value=paragraph.replace(/\s+/g," ").trim();
        if(!/\bmit(?:\s+license)?\b/iu.test(value))continue;
        const unequivocalAbsence=/(?:патентн\p{L}*\s+прав\p{L}*).{0,65}(?:не\s+(?:охраня\p{L}*|защища\p{L}*|существ\p{L}*|возника\p{L}*)|отсутств\p{L}*\s+(?:вообще|полностью))|(?:no|without)\s+(?:patent\s+rights|patent\s+protection)/iu.test(value);
        const oneParagraph=/(?:текст\s+лицензи\p{L}*|mit(?:\s+license)?).{0,75}состо\p{L}*\s+из\s+(?:одного|1)\s+абзац\p{L}*/iu.test(value);
        if(unequivocalAbsence || oneParagraph) {
            return {
                valid:false,
                shouldRetry:true,
                reason:unequivocalAbsence
                    ? "Отсутствие явного патентного гранта MIT не доказывает отсутствие или незащищённость всех патентных прав. Исправь категорический вывод."
                    : "MIT License не состоит из одного абзаца: отдельно представлены грант разрешений, условие уведомлений и отказ от гарантий. Исправь описание структуры."
            };
        }
    }
    return null;
}
