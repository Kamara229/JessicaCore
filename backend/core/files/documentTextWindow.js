import { createHash } from "node:crypto";
/** A bounded, lossless page of extracted document text (UTF-16 offsets). */
export const DOCUMENT_WINDOW_CHARS = 16000;

export function parseDocumentTextOffset(value) {
    if (value === undefined || value === null || value === "") return 0;
    const raw = String(value);
    if (!/^(0|[1-9][0-9]*)$/.test(raw)) return null;
    const offset = Number(raw);
    return Number.isSafeInteger(offset) ? offset : null;
}

function splitsSurrogatePair(text, index) {
    if (index <= 0 || index >= text.length) return false;
    const before = text.charCodeAt(index - 1);
    const after = text.charCodeAt(index);
    return before >= 0xd800 && before <= 0xdbff &&
        after >= 0xdc00 && after <= 0xdfff;
}

/** A later page can be retrieved by resending the same file with nextOffset. */
export function pageDocumentText(complete, offset = 0) {
    if (typeof complete !== "string" || !Number.isSafeInteger(offset) ||
        offset < 0 || offset > complete.length ||
        splitsSurrogatePair(complete, offset)) return null;

    let end = Math.min(offset + DOCUMENT_WINDOW_CHARS, complete.length);
    if (splitsSurrogatePair(complete, end)) end--;
    const hasMore = end < complete.length;
    return {
        text: complete.slice(offset, end),
        truncated: complete.length > DOCUMENT_WINDOW_CHARS,
        extractedChars: complete.length,
        offset,
        nextOffset: hasMore ? end : null,
        hasMore,
        sourceSha256: createHash("sha256").update(complete, "utf8").digest("hex")
    };
}
