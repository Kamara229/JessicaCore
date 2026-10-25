import {assembleDocumentPages} from "./documentPageAssembly.js";
import {planDocumentSectionWindows} from "./documentSectionReadPlan.js";
import {beginWindowLedger} from "./documentWindowLedger.js";
import {analyzePagedLongDocument} from "./documentPagedAnalysis.js";
import {identifyDocumentAnswer} from "./documentResponseIdentity.js";
import {scopeCitedDocumentFragments} from "./documentCitationSpans.js";
import {scopeVerifiedClaimRecords,compareVerifiedDocumentClaims} from "./documentClaimComparison.js";

const MAX_FILES=3;
const MAX_WINDOWS=16;

function reject(stage,reason){
    return {success:false,mode:"document",stage,
        text:reason,readingCoverage:{complete:false,completeSourceAnalysisConfirmed:false}};
}

/** Preflight EVERY document before consuming AI quota or claiming any coverage. */
export function planMultiplePagedDocuments(attachments) {
    if(!Array.isArray(attachments)||attachments.length<2||
       attachments.length>MAX_FILES)return null;
    const files=[];
    let totalWindows=0;
    for(let i=0;i<attachments.length;i++){
        const item=attachments[i];
        const pages=item?.pages;
        const name=String(item?.name||"").trim();
        if(!name||!Array.isArray(pages)||!pages.length)return null;
        const hash=pages[0]?.sourceSha256;
        if(!/^[a-f0-9]{64}$/u.test(String(hash||""))||
           pages.some(page=>page.name!==undefined&&page.name!==name))
            return null;
        const source=assembleDocumentPages(pages);
        if(!source?.complete)return null;
        const plan=planDocumentSectionWindows(source.text);
        if(!plan.complete)return null;
        if(!beginWindowLedger(plan,hash))return null;
        totalWindows+=plan.windows.length;
        if(totalWindows>MAX_WINDOWS)return null;
        files.push({index:i+1,name,sourceSha256:hash,
            sourceChars:source.text.length,plannedWindows:plan.windows.length});
    }
    return {files,totalWindows};
}

/** Preserve per-file identities, structure and source spans across all attachments. */
export async function analyzeMultiplePagedDocuments({task,attachments,generate}) {
    const plan=planMultiplePagedDocuments(attachments);
    if(!plan)return reject("document-multi-pages",
        "Не удалось подтвердить полный текст всех файлов или превышен лимит окон. "+
        "Отправь документы по одному; частичный анализ не выдаётся за полный.");
    const summaries=[],sections=[],citationSpans=[],verifiedClaims=[];
    let kept=0,rejected=0,windowsDone=0;
    for(const info of plan.files){
        const result=await analyzePagedLongDocument({
            task,attachment:attachments[info.index-1],generate,embedReceipt:false
        });
        const coverage=result?.readingCoverage;
        if(result?.success!==true||coverage?.allWindowsPresented!==true||
           coverage?.sourceSha256!==info.sourceSha256||
           coverage?.sourceChars!==info.sourceChars||
           coverage?.processedWindows!==info.plannedWindows||
           result?.sectionCoverage?.allSectionsProcessed!==true){
            return {success:false,mode:"document",
                stage:result?.stage||"document-multi-coverage",
                text:"Анализ нескольких документов не завершён. "+
                    "Нельзя утверждать, что все файлы прочитаны.",
                readingCoverage:{plannedWindows:plan.totalWindows,
                    processedWindows:windowsDone+
                        (Number.isSafeInteger(coverage?.processedWindows)?coverage.processedWindows:0),
                    files:summaries,
                    incompleteFile:coverage?{...coverage,fileIndex:info.index,name:info.name}:null,
                    complete:false,allWindowsPresented:false,
                    completeSourceAnalysisConfirmed:false}};
        }
        kept+=result.groundedClaims;
        rejected+=result.unsupportedClaimsDropped;
        windowsDone+=coverage.processedWindows;
        const scoped=result.text.replace(/\[О(\d+):Ф1:Ф(\d+)\]/gu,
            (_m,w,f)=>`[Д${info.index}:О${w}:Ф1:Ф${f}]`);
        const scopedSpans=scopeCitedDocumentFragments(result.citationSpans,info.index);
        if(!scopedSpans||!scopedSpans.length)return reject("document-multi-spans",
            "Не удалось привязать цитаты к исходному файлу.");
        const scopedClaims=scopeVerifiedClaimRecords(
            result.verifiedClaims,info.index,scopedSpans);
        if(!scopedClaims)return reject("document-multi-claims",
            "Не удалось связать выводы с цитатами каждого файла.");
        citationSpans.push(...scopedSpans);
        verifiedClaims.push(...scopedClaims);
        sections.push(`Документ ${info.index}: ${info.name}\n${scoped}`);
        summaries.push({...coverage,fileIndex:info.index,name:info.name,
            sectionCoverage:result.sectionCoverage});
    }
    const body=sections.join("\n\n")+
        "\n\nПроверка: все подтверждённые страницы были переданы на анализ. "+
        "Цитаты сверены с текстом, но достоверность научных и правовых выводов отдельно не установлена.";
    const claimComparison=compareVerifiedDocumentClaims(verifiedClaims,citationSpans);
    if(!claimComparison)return reject("document-multi-comparison",
        "Межфайловые выводы не удалось сопоставить с проверенными ссылками.");
    const audit=identifyDocumentAnswer(body,kept,rejected);
    return {success:true,mode:"document",validated:false,evidenceChecked:true,
        groundedClaims:kept,unsupportedClaimsDropped:rejected,
        readingCoverage:{plannedWindows:plan.totalWindows,
            processedWindows:windowsDone,files:summaries,
            allWindowsPresented:windowsDone===plan.totalWindows,
            exactTextCoverageConfirmed:true,completeSourceAnalysisConfirmed:false},
        citationSpans,verifiedClaims,claimComparison,
        responseId:audit.id,responseSha256:audit.sha256,text:audit.text};
}
