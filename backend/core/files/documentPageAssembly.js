import { createHash } from "node:crypto";
import { DOCUMENT_WINDOW_CHARS } from "./documentTextWindow.js";

const MAX_PAGES = 256;
const MAX_CHARS = 4_000_000;
const digest = text => createHash("sha256").update(text, "utf8").digest("hex");

/** Reassemble only a complete, contiguous series of pages for one document. */
export function assembleDocumentPages(pages) {
    if (!Array.isArray(pages) || !pages.length || pages.length > MAX_PAGES)
        return null;
    const first = pages[0];
    const total = first?.extractedChars;
    const name = first?.name;
    const expectedDigest = first?.sourceSha256;
    if (!Number.isSafeInteger(total) || total < 1 || total > MAX_CHARS ||
        (expectedDigest !== undefined && !/^[a-f0-9]{64}$/.test(expectedDigest)))
        return null;

    let end = 0;
    const texts = [];
    for (let i = 0; i < pages.length; i++) {
        const page = pages[i];
        if (!page || typeof page.text !== "string" || !page.text.length ||
            page.text.length > DOCUMENT_WINDOW_CHARS ||
            page.extractedChars !== total ||
            page.offset !== end ||
            page.hasMore !== (i < pages.length - 1) ||
            (name !== undefined && page.name !== name) ||
            page.sourceSha256 !== expectedDigest) return null;
        const next = end + page.text.length;
        if (next > total || (page.hasMore ? page.nextOffset !== next :
            page.nextOffset !== null || next !== total)) return null;
        // A page boundary must not split a Unicode surrogate pair.
        if (i && /[\uDC00-\uDFFF]/u.test(page.text[0]) &&
            /[\uD800-\uDBFF]/u.test(texts[i - 1].at(-1))) return null;
        end = next;
        texts.push(page.text);
    }
    if (end !== total) return null;
    const text = texts.join("");
    if (expectedDigest && digest(text) !== expectedDigest) return null;
    return { text, extractedChars: total, complete: true, pageCount: pages.length };
}
