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
import {analyzeMultiplePagedDocuments} from "./documentMultiPageAnalysis.js";
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
    if(Array.isArray(attachments)&&attachments.length>1&&
       attachments.some(item=>Array.isArray(item?.pages))){
        if(!String(task||"").trim())return errorResult("Укажи задачу для документов.");
        if(getAIProviderAvailability().blocked)
            return errorResult("Достигнут лимит ИИ.","ai-provider-cooldown");
        try{return await analyzeMultiplePagedDocuments({task,attachments,generate});}
        catch(error){
            if(error?.status===429){
                registerAIProviderLimit(error);
                return errorResult("Лимит ИИ: анализ прерван.","ai-provider-cooldown");
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
                return errorResult("Лимит ИИ: анализ прерван.","ai-provider-cooldown");
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
