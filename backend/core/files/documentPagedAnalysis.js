import {assembleDocumentPages} from "./documentPageAssembly.js";
import {createDocumentEvidence} from "./documentEvidence.js";
import {planDocumentSectionWindows} from "./documentSectionReadPlan.js";
import {validateAndRenderDocumentAnswer} from "./documentAnswerContract.js";
import {identifyDocumentAnswer} from "./documentResponseIdentity.js";

/** Use only cryptographically verified complete pages; fail closed on gaps. */
export async function analyzePagedLongDocument({task,attachment,generate}) {
    const name=String(attachment?.name||"Документ");
    const pages=attachment?.pages;
    const hash=pages?.[0]?.sourceSha256;
    const valid=typeof hash==="string"&&/^[a-f\d]{64}$/u.test(hash)&&
        (!pages[0].name||pages[0].name===name);
    const assembled=valid?assembleDocumentPages(pages):null;
    if(!assembled)return {success:false,mode:"document",stage:"document-pages",
        text:"Страницы документа не прошли проверку целостности."};
    const plan=planDocumentSectionWindows(assembled.text);
    if(!plan.complete)return {success:false,mode:"document",stage:"document-read-plan",
        text:"Полный анализ требует продолжения по частям. Не подтверждена полнота.",
        readingCoverage:{plannedWindows:plan.windows.length,processedWindows:0,
            complete:false,reason:plan.reason,sourceChars:assembled.text.length}};
    let processed=0,kept=0,rejected=0;
    const answers=[];
    for(const window of plan.windows){
        const excerpt={name,text:window.text,truncated:false};
        const bundle=createDocumentEvidence([excerpt]);
        const prompt=[
            "ЗАДАНИЕ:",String(task).trim().slice(0,3000),
            "СОДЕРЖИМОЕ ТОЛЬКО ОДНОГО ОКНА:",
            `Окно ${window.number}, раздел «${window.title}», символы ${window.startOffset}–${window.endOffset}.`,
            bundle.promptText,
            "Верни JSON с проверяемыми цитатами. Не заявляй, что анализировал другие окна."
        ].join("\n");
        let checked;
        try {
            checked=validateAndRenderDocumentAnswer(
                await generate(prompt,task,[excerpt]),bundle,task);
        }catch(error){
            if(error?.status===429)throw error;
            return {success:false,mode:"document",stage:"document-window-error",
                text:"Окно не обработано; полный анализ не завершён.",
                readingCoverage:{plannedWindows:plan.windows.length,processedWindows:processed,complete:false}};
        }
        if(!checked.success)return {success:false,mode:"document",stage:"document-window-evidence",
            text:"Для одного из окон не подтверждены цитаты. Полный анализ не завершён.",
            readingCoverage:{plannedWindows:plan.windows.length,processedWindows:processed,complete:false}};
        processed++;kept+=checked.kept;rejected+=checked.rejected;
        const scoped=checked.text.replace(/\[Ф1:Ф(\d+)\]/gu,
            (_whole,n)=>`[О${window.number}:Ф1:Ф${n}]`);
        answers.push(`Окно ${window.number} — ${window.title} (${window.startOffset}–${window.endOffset})\n${scoped}`);
    }
    const body=answers.join("\n\n")+
        "\n\nГраницы проверки: все окна переданы для анализа и цитаты сверены. " +
        "Это не доказывает научную или юридическую правильность выводов.";
    const audit=identifyDocumentAnswer(body,kept,rejected);
    return {success:true,mode:"document",validated:false,evidenceChecked:true,
        groundedClaims:kept,unsupportedClaimsDropped:rejected,
        readingCoverage:{plannedWindows:plan.windows.length,processedWindows:processed,
            allWindowsPresented:true,completeSourceAnalysisConfirmed:false,
            sourceSha256:hash,sourceChars:assembled.text.length},
        responseId:audit.id,responseSha256:audit.sha256,text:audit.text};
}
