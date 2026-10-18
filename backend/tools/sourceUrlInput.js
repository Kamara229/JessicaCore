/*
 * Trim invisible chat-copy artefacts only at a URL boundary.
 * Do not change the path or invent a replacement source.
 */
export function sanitizeSourceUrlInput(value) {
    return String(value || "").trim()
        .replace(/[\u200b-\u200f\u2060\ufeff\ufffd]+$/giu, "")
        .replace(/[.,;!?]+$/g, "");
}
