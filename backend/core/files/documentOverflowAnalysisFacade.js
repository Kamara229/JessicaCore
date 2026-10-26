import {
    getAIProviderAvailability, registerAIProviderLimit
} from "../../ai/providerAvailability.js";
import {assembleDocumentPages} from "./documentPageAssembly.js";
import {verifyMaterializedOverflowBatches} from "./documentOverflowBatchVerification.js";
import {planVerifiedDocumentAnalysisStages} from "./documentAnalysisStagePlan.js";
import {MAX_VERIFIED_DOCUMENT_WINDOWS} from "./documentVerifiedWindowLimit.js";
import {analyzeVerifiedDocumentStage} from "./documentStageRunner.js";
import {checkStageReportOrder} from "./documentStageReportOrder.js";
import {mergeAuthenticatedDocumentStages} from "./documentStageAuthenticatedMerge.js";
import {authenticatedStageResumeStatus} from "./documentStageResumeStatus.js";
import {analyzePagedLongDocument} from "./documentPagedAnalysis.js";
import {addInterruptedDocumentCoverage} from "./documentQuotaCoverage.js";

const errorResult=(text,stage="document-analysis")=>
    ({success:false,stage,mode:"document",text});

/** Narrow facade for verified oversized-document batches and stage reports. */
export async function analyzeOverflowDocumentAttachments({task,attachments},generate){
    // Source windows may span many batches, but no prefix is an AI result.
    if(Array.isArray(attachments)&&attachments.some(item=>
        item && typeof item==="object"&&Object.hasOwn(item,"overflowBatches"))){
        if(attachments.length!==1||!Array.isArray(attachments[0]?.pages))
            return errorResult("Ожидается один полный документ со всеми страницами.",
                "document-overflow-batches-invalid");
        const item=attachments[0];
        const complete=assembleDocumentPages(item.pages);
        if(!complete?.complete||!String(item.name||"")||
           item.pages.some(page=>page.name!==item.name))
            return errorResult("Не подтверждена целостность исходного файла.",
                "document-overflow-batches-invalid");
        const checked=verifyMaterializedOverflowBatches(complete.text,
            item.pages[0].sourceSha256,item.overflowBatches);
        if(!checked)return errorResult("Нарушен порядок окон или контрольная сумма.",
            "document-overflow-batches-invalid");
        const stagePlan=checked.complete&&
            checked.totalWindows>MAX_VERIFIED_DOCUMENT_WINDOWS?
            planVerifiedDocumentAnalysisStages(complete.text,checked):null;
        if(checked.complete&&checked.totalWindows>MAX_VERIFIED_DOCUMENT_WINDOWS&&
           !stagePlan)
            return errorResult("Не удалось построить проверяемые этапы анализа.",
                "document-overflow-stage-plan-invalid");
        // Reports received from a client are untrusted metadata, not model proof.
        if(Object.hasOwn(item,"stageReports")){
            if(Object.hasOwn(item,"stageNumber")||!checked.complete||!stagePlan)
                return errorResult("Серия этапов или исходный файл неполны.",
                    "document-stage-reports-invalid");
            const coordinates=checkStageReportOrder(complete.text,checked,
                item.stageReports,task);
            if(!coordinates)return errorResult(
                "Несовпадение порядка этапов или смещений.",
                "document-stage-reports-invalid");
            if(coordinates.allStageReportsReceived&&
               coordinates.reportOriginAuthenticated){
                const consolidated=mergeAuthenticatedDocumentStages(
                    coordinates,item.stageReports);
                if(consolidated)return consolidated;
                return errorResult("Не удалось объединить подтверждённые этапы.",
                    "document-stage-merge-invalid");
            }
            const resume=authenticatedStageResumeStatus(coordinates);
            if(resume)return resume;
            return {success:false,mode:"document",
                stage:"document-stage-reports-untrusted",
                text:"Порядок этапов проверен; происхождение ответов и полнота анализа не подтверждены.",
                readingCoverage:{...coordinates,processedWindows:0,
                    complete:false,completeSourceAnalysisConfirmed:false}};
        }
        if(Object.hasOwn(item,"stageNumber")){
            if(!checked.complete||!stagePlan)
                return errorResult("Нет полного проверенного плана этапов.",
                    "document-stage-input");
            if(getAIProviderAvailability().blocked)
                return errorResult("Достигнут лимит ИИ.","ai-provider-cooldown");
            try {
                return await analyzeVerifiedDocumentStage({
                    task,source:complete.text,verified:checked,
                    stageNumber:item.stageNumber,name:item.name
                },generate);
            }catch(error){
                if(error?.status===429){
                    registerAIProviderLimit(error);
                    return addInterruptedDocumentCoverage(
                        errorResult("Лимит ИИ: этап анализа прерван.",
                            "ai-provider-cooldown"),error);
                }
                return errorResult("Этап анализа не завершён.",
                    "document-stage-error");
            }
        }
        if(!checked.complete||checked.totalWindows>MAX_VERIFIED_DOCUMENT_WINDOWS)
            return {success:false,mode:"document",
                stage:checked.complete?"document-overflow-analysis-pending":
                    "document-overflow-batches-incomplete",
                text:checked.complete?
                    "Свыше 32 окон: нужен этапный анализ с сохранением результатов.":
                    "Не все части переданы. Анализ не запускался.",
                readingCoverage:{complete:false,
                    plannedWindows:checked.totalWindows,processedWindows:0,
                    allSourceWindowsVerified:checked.allSourceWindowsVerified,
                    sourceChars:checked.sourceChars,
                    sourceSha256:checked.sourceSha256,
                    collectedChars:checked.collectedChars,
                    nextBatchNumber:checked.nextBatchNumber,
                    ...(stagePlan?{stagePlan}:{}),
                    completeSourceAnalysisConfirmed:false}};
        if(!String(task||"").trim())
            return errorResult("Укажи задачу для документа.");
        if(getAIProviderAvailability().blocked)
            return errorResult("Достигнут лимит ИИ.","ai-provider-cooldown");
        try {
            return await analyzePagedLongDocument({
                task,attachment:item,generate,verifiedOverflow:checked
            });
        }catch(error){
            if(error?.status===429){
                registerAIProviderLimit(error);
                return addInterruptedDocumentCoverage(
                    errorResult("Лимит ИИ: анализ прерван.","ai-provider-cooldown"),
                    error);
            }
            return errorResult("Не удалось завершить пораздельный анализ.",
                "document-overflow-error");
        }
    }
    return errorResult("Неверный формат частей документа.",
        "document-overflow-batches-invalid");
}
