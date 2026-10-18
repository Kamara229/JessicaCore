import { getGroqClient } from "../../ai/groqClient.js";
import {
    getAIProviderAvailability,
    registerAIProviderLimit
} from "../../ai/providerAvailability.js";
import { createDocumentEvidence } from "./documentEvidence.js";
import { identifyDocumentAnswer } from "./documentResponseIdentity.js";
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
        bundle.partial
            ? "ВНИМАНИЕ: часть документа обрезана по лимитам извлечения; нельзя делать выводы о непрочитанном."
            : "Все доступные извлечённые фрагменты приведены выше.",
        "Проверь факты по дословным цитатам. Не выводи факт из имени файла."
    ].join("\n");
}

async function generateDocumentAnswer(prompt) {
    const client = getGroqClient();
    const answer = await client.chat.completions.create({
        model: "openai/gpt-oss-20b",
        temperature: 0,
        reasoning_effort: "low",
        include_reasoning: false,
        max_completion_tokens: 2200,
        response_format: { type: "json_object" },
        messages: [
            { role: "system", content: documentAnalysisInstructions() },
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
        const raw = await generate(buildDocumentPrompt({task, attachments: bundle.docs}));
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
