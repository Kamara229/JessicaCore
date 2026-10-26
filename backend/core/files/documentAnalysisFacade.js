import { getGroqClient } from "../../ai/groqClient.js";
import {buildProgrammingFoundation} from "../foundations/programmingPolicy.js";
import {buildDataFoundation} from "../foundations/dataFoundation.js";
import {profileStructuredAttachments} from "../foundations/dataProfile.js";
import {
    getAIProviderAvailability,
    registerAIProviderLimit
} from "../../ai/providerAvailability.js";
import { createDocumentEvidence } from "./documentEvidence.js";
import { identifyDocumentAnswer } from "./documentResponseIdentity.js";
import {summarizeDocumentCitationCoverage} from "./documentCitationCoverage.js";
import {analyzePagedLongDocument} from "./documentPagedAnalysis.js";
import {MAX_VERIFIED_DOCUMENT_WINDOWS} from "./documentVerifiedWindowLimit.js";
import {assembleDocumentChunks} from "./documentChunkAssembly.js";
import {assembleDocumentPages} from "./documentPageAssembly.js";
import {verifyMaterializedOverflowBatches} from "./documentOverflowBatchVerification.js";
import {planVerifiedDocumentAnalysisStages} from "./documentAnalysisStagePlan.js";
import {analyzeVerifiedDocumentStage} from "./documentStageRunner.js";
import {checkStageReportOrder} from "./documentStageReportOrder.js";
import {analyzeMultiplePagedDocuments} from "./documentMultiPageAnalysis.js";
import {addInterruptedDocumentCoverage} from "./documentQuotaCoverage.js";
import {
    documentAnalysisInstructions,
    validateAndRenderDocumentAnswer
} from "./documentAnswerContract.js";

/**
 * File-first analysis. Documents are already evidence; the public web-search
 * Planner must not be used to rediscover an uploaded file by its filename.
 * Document-specific content never becomes a reusable Experience candidate.
 */
export function buildDocumentPrompt({ task, attachments }) {
    const bundle = createDocumentEvidence(attachments);
    return [
        "ТЕКУЩЕЕ ЗАДАНИЕ ПОЛЬЗОВАТЕЛЯ:",
        String(task || "").trim().slice(0, 3000),
        "",
        "ПРИКРЕПЛЁННЫЕ ДОКУМЕНТЫ (все фрагменты помечены стабильными ID):",
        bundle.promptText,
        "",
        "DETERMINISTIC STRUCTURE PROFILE (metadata only; not proof of semantic correctness or full-file coverage):",
        JSON.stringify(profileStructuredAttachments(bundle.docs)),
        "",
        bundle.partial
            ? "ВНИМАНИЕ: часть документа обрезана по лимитам извлечения; нельзя делать выводы о непрочитанном."
            : "Все доступные извлечённые фрагменты приведены выше.",
        "Проверь факты по дословным цитатам. Не выводи факт из имени файла."
    ].join("\n");
}

async function generateDocumentAnswer(prompt,task,attachments) {
    const client = getGroqClient();
    const answer = await client.chat.completions.create({
        model: "openai/gpt-oss-20b",
        temperature: 0,
        reasoning_effort: "low",
        include_reasoning: false,
        max_completion_tokens: 2200,
        response_format: { type: "json_object" },
        messages: [
            { role: "system", content: documentAnalysisInstructions() + "\n" + buildProgrammingFoundation(task,{files:attachments,stage:"document"}) + "\n" + buildDataFoundation(task,{files:attachments,stage:"document"}) },
            { role: "user", content: prompt }
        ]
    });
    return String(answer?.choices?.[0]?.message?.content || "").trim();
}

function errorResult(message, stage = "document-analysis") {
    return { success: false, stage, mode: "document", text: message };
}

export async function analyzeDocumentAttachments(
    { task, attachments },
    generate = generateDocumentAnswer
) {
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
                item.stageReports);
            if(!coordinates)return errorResult(
                "Несовпадение порядка этапов или смещений.",
                "document-stage-reports-invalid");
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
    // Accept chunked extraction only after complete, verified SHA-256
    // reassembly. Never downgrade an incomplete chunk to loose text.
    if(Array.isArray(attachments)&&attachments.some(item=>
        item!==null&&typeof item==="object"&&Object.hasOwn(item,"chunks"))){
        if(attachments.length>3)return errorResult(
            "Слишком много файлов для проверки частей.","document-chunks-invalid");
        const normalized=[];
        for(const item of attachments){
            if(item?.chunks===undefined){
                if(item?.pages)normalized.push(item);
                else return errorResult("Некорректный набор документов.",
                    "document-chunks-invalid");
                continue;
            }
            const assembled=assembleDocumentChunks(item.chunks);
            if(!assembled)return errorResult("Части документа нарушают порядок или целостность.",
                "document-chunks-invalid");
            if(!assembled.complete)return {success:false,mode:"document",
                stage:"document-chunks-incomplete",
                text:"Получены не все части документа. Анализ не запускался.",
                readingCoverage:{complete:false,sourceIntegrityVerified:false,
                    processedWindows:0,collectedChars:assembled.collectedChars,
                    sourceChars:assembled.extractedChars,
                    nextOffset:assembled.nextOffset,
                    completeSourceAnalysisConfirmed:false}};
            if(item.name!==assembled.name)return errorResult(
                "Имя файла не совпало с извлечёнными частями.","document-chunks-invalid");
            normalized.push({name:assembled.name,pages:assembled.pages});
        }
        return analyzeDocumentAttachments({task,attachments:normalized},generate);
    }
    if(Array.isArray(attachments)&&attachments.length>1&&
       attachments.some(item=>Array.isArray(item?.pages))){
        if(!String(task||"").trim())return errorResult("Укажи задачу для документов.");
        if(getAIProviderAvailability().blocked)
            return errorResult("Достигнут лимит ИИ.","ai-provider-cooldown");
        try{return await analyzeMultiplePagedDocuments({task,attachments,generate});}
        catch(error){
            if(error?.status===429){
                registerAIProviderLimit(error);
                return addInterruptedDocumentCoverage(
                    errorResult("Лимит ИИ: анализ прерван.","ai-provider-cooldown"),
                    error);
            }
            return errorResult("Проверка документов не завершена.","document-multi-error");
        }
    }
    if (Array.isArray(attachments)&&attachments.length===1&&
        Array.isArray(attachments[0]?.pages)) {
        if (!String(task||"").trim()) return errorResult("Укажи задачу для документа.");
        if (getAIProviderAvailability().blocked)
            return errorResult("Достигнут лимит ИИ.","ai-provider-cooldown");
        try { return await analyzePagedLongDocument({task,attachment:attachments[0],generate}); }
        catch(error){
            if(error?.status===429){
                registerAIProviderLimit(error);
                return addInterruptedDocumentCoverage(
                    errorResult("Лимит ИИ: анализ прерван.","ai-provider-cooldown"),
                    error);
            }
            return errorResult("Анализ документа не завершён.","document-window-error");
        }
    }
    const bundle = createDocumentEvidence(attachments);
    if (!bundle.docs.length || !bundle.fragments.length) {
        return errorResult("Прикреплённый документ не содержит доступного текста. Проверь файл и отправь его снова.");
    }
    if (!String(task || "").trim()) {
        return errorResult("Укажи, что необходимо сделать с документом.");
    }
    console.log("Jessica document analysis started:", {
        fileCount: bundle.docs.length,
        totalChars: bundle.docs.reduce((sum, d) => sum + d.text.length, 0),
        fragmentCount: bundle.fragments.length,
        partial: bundle.partial
    });

    if (getAIProviderAvailability().blocked) {
        return errorResult(
            "Документ успешно прочитан, но сейчас ИИ временно недоступен из-за лимита запросов. Повтори анализ позже; файл остался в чате.",
            "ai-provider-cooldown"
        );
    }
    try {
        const raw = await generate(buildDocumentPrompt({task, attachments: bundle.docs}),task,bundle.docs);
        const checked = validateAndRenderDocumentAnswer(raw, bundle);
        if (!checked.success) {
            console.warn("Jessica document evidence rejected:", {
                reason: checked.reason,
                kept: checked.kept,
                rejected: checked.rejected
            });
            return errorResult(
                "Документ прочитан, но не удалось сформировать ответ с подтверждёнными цитатами. " +
                "Я не буду выдавать непроверенные утверждения за содержание файла. Повтори запрос позже.",
                "document-evidence"
            );
        }
        const citationCoverage = summarizeDocumentCitationCoverage(bundle,checked.citedRefs);
        const audit = identifyDocumentAnswer(checked.text, checked.kept, checked.rejected);
        console.log("Jessica document analysis completed:", {
            fileCount: bundle.docs.length,
            groundedClaims: checked.kept,
            unsupportedClaimsDropped: checked.rejected,
            answerChars: checked.text.length,
            responseId: audit.id,
            responseSha256: audit.sha256
        });
        return {
            success: true,
            mode: "document",
            validated: false,
            evidenceChecked: true,
            groundedClaims: checked.kept,
            unsupportedClaimsDropped: checked.rejected,
            citationCoverage,
            responseId: audit.id,
            responseSha256: audit.sha256,
            text: audit.text
        };
    } catch (error) {
        if (error?.status === 429) {
            registerAIProviderLimit(error);
            return errorResult(
                "Документ прочитан, но достигнут временный лимит ИИ. Повтори анализ позже — вложение осталось в чате.",
                "ai-provider-cooldown"
            );
        }
        console.error("Jessica document analysis error:", {
            status: error?.status || null,
            code: error?.code || null
        });
        return errorResult("Не удалось завершить анализ документа. Файл сохранён в чате; повтори запрос позже.");
    }
}
