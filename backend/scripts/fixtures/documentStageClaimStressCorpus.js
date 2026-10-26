import {createHash} from "node:crypto";

export function makeDocumentStressSource(sectionCount=128){
    if(!Number.isSafeInteger(sectionCount)||sectionCount<1||
       sectionCount>128)return null;
    const sections=[];
    for(let n=1;n<=sectionCount;n++)
        sections.push("Раздел "+n+"\n"+"а".repeat(16000)+"\n");
    return sections.join("");
}

/** Deterministic equal-length, distinct 256-window inputs per request. */
export function makeConcurrentStressSource(id){
    if(!Number.isSafeInteger(id)||id<1||id>10)return null;
    const marker=("JESSICA"+String(id)).padEnd(32,"я");
    return makeDocumentStressSource().replace("а".repeat(32),marker);
}

export function hashDocumentStressSource(source){
    return createHash("sha256").update(source,"utf8").digest("hex");
}

export function claimsFromStressWindows(windows){
    if(!Array.isArray(windows)||windows.length!==256)return null;
    return windows.map((window,index)=>{
        const n=index+1;
        const statement=n===1?"Договор действует":
            n===256?"Договор не действует":"Анализ норм";
        return {
            stageNumber:Math.ceil(n/16),
            windowNumber:n,
            sectionIndex:window.sectionIndex,
            startOffset:window.startOffset,endOffset:window.endOffset,
            claim:"• Позиция автора: "+statement+
                " [О"+n+":Ф1:Ф1]",
            evidence:"  Основание: «ааааа»"
        };
    });
}
