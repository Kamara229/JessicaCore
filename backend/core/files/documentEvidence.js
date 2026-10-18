import { normalizeAttachmentContexts } from "./attachmentContext.js";

const MAX_FRAGMENT_CHARS = 900;
const MAX_FRAGMENTS = 100;

function splitLongParagraph(text) {
    const chunks = [];
    let rest = text;
    while (rest.length > MAX_FRAGMENT_CHARS) {
        const boundary = rest.lastIndexOf(" ", MAX_FRAGMENT_CHARS);
        const end = boundary > MAX_FRAGMENT_CHARS / 2 ? boundary : MAX_FRAGMENT_CHARS;
        chunks.push(rest.slice(0, end).trim());
        rest = rest.slice(end).trim();
    }
    if (rest) chunks.push(rest);
    return chunks;
}

/**
 * Every source fragment gets a stable, unique reference for this request.
 * The original text is never replaced with a paraphrase before the model sees it.
 */
export function createDocumentEvidence(attachments) {
    const docs = normalizeAttachmentContexts(attachments);
    const fragments = [];
    const index = new Map();
    let omitted = 0;
    for (let fileNumber = 0; fileNumber < docs.length; fileNumber++) {
        const document = docs[fileNumber];
        let part = 0;
        const paragraphs = document.text
            .split(/\n\s*\n/g)
            .flatMap(value => splitLongParagraph(value.trim()))
            .filter(Boolean);
        // Keep all text, even if the extractor did not preserve blank lines.
        for (const paragraph of paragraphs) {
            if (fragments.length >= MAX_FRAGMENTS) {
                omitted++;
                continue;
            }
            const ref = `Ф${fileNumber + 1}:Ф${++part}`;
            const fragment = { ref, text: paragraph, name: document.name };
            fragments.push(fragment);
            index.set(ref, fragment);
        }
    }
    return {
        docs,
        fragments,
        index,
        partial: omitted > 0 || docs.some(d => d.truncated),
        promptText: fragments.map(fragment =>
            `[${fragment.ref}] (${fragment.name})\n${fragment.text}`
        ).join("\n\n"),
        omitted
    };
}
