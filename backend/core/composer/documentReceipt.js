/*
 * A deterministic receipt for a fetch-only subtask.
 * It proves that the expected document was retrieved; it does NOT
 * claim its legal terms were understood or compared.
 */
import { sanitizeSourceUrlInput } from "../../tools/sourceUrlInput.js";

function canonicalUrl(input) {
    try {
        const url=new URL(sanitizeSourceUrlInput(input));
        if (!["http:","https:"].includes(url.protocol)) return "";
        return url.origin + url.pathname.replace(/\/+$/g,"").toLowerCase();
    } catch { return ""; }
}

export function getDocumentReceipt(task, plan, taskRunResult) {
    if (plan?.intent !== "native_document_capture") return null;
    const results=taskRunResult?.results;
    if (!Array.isArray(results) || results.length !== 1) return null;
    const fetch=results[0];
    const expected=plan?.steps?.[0]?.arguments?.url;
    const data=fetch?.data;
    const content=typeof data?.content === "string" ? data.content.trim() : "";

    if (fetch?.tool !== "web_fetch" || fetch?.success !== true
        || !content || content.length < 350 || data?.truncated === true
        || !canonicalUrl(expected)
        || canonicalUrl(expected) !== canonicalUrl(data?.url)) {
        return null;
    }

    const lowerTask=String(task || "").toLowerCase();
    if (lowerTask.includes("mit license")
        && (!/permission is hereby granted/i.test(content)
            || !/mit license/i.test(content))) return null;

    if (lowerTask.includes("apache license")
        && (!/apache license/i.test(content)
            || !/grant of patent license/i.test(content))) return null;

    const url=String(data.url);
    const title=String(data.title || "").trim();
    return {
        success: true,
        source: "document-receipt",
        url,
        length: content.length,
        text: "Получен текст документа: " + (title || url)
            + "\nИсточник: " + url
            + "\nРазмер: " + content.length + " символов.\n"
            + "Фрагмент оригинального текста (не интерпретация):\n"
            + content.slice(0, 3500)
    };
}
