import { sanitizeSourceUrlInput } from "../../tools/sourceUrlInput.js";

/* Fetch-only subtasks need no AI planning. Never treat a comparison
 * or synthesis request as a fetch-only task. */
export function buildNativeDocumentFetchPlan(task) {
    const text=String(task || "").trim();
    if (!/^(?:собрать|получить|загрузить)\s+(?:полный\s+)?текст(?=\s|$)/iu.test(text)
        && !/^(?:fetch|retrieve|download)\s+(?:the\s+)?(?:full\s+)?(?:text|document)\b/i.test(text)) {
        return null;
    }
    const links=text.match(/https?:\/\/[^\s<>"')\]]+/gi) || [];
    if (links.length !== 1) return null;
    const url=sanitizeSourceUrlInput(links[0]);
    try {
        const parsed=new URL(url);
        if (!["https:","http:"].includes(parsed.protocol)) return null;
    } catch { return null; }

    return {
        intent:"native_document_capture",
        requiresTools:true,
        reasoningSummary:"Сохранить документ из предоставленного URL.",
        evidence:{mode:"source_content",reason:"Подтвердить содержимое исходного документа."},
        steps:[{id:"native_document_fetch",tool:"web_fetch",arguments:{url}}],
        metadata:{plannerVersion:"native-document-fetch-v1",generatedAt:new Date().toISOString()}
    };
}
