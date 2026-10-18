import { createHash, randomBytes } from "node:crypto";

/**
 * Request-local identity for document answers. The same ID is printed in the
 * answer shown on Android and logged when /api/solve finishes returning it.
 * No document contents, quotations or user task text are written to logs.
 */
export function identifyDocumentAnswer(text, kept, rejected) {
    const id = "JDOC-" + randomBytes(5).toString("hex").toUpperCase();
    const summary = [
        "Проверка источников Jessica:",
        `${kept} утверждений с подтверждёнными цитатами,`,
        `${rejected} отклонено.`,
        `Код ответа: ${id}`
    ].join(" ");
    const fullText = String(text || "") + "\n\n" + summary;
    const sha256 = createHash("sha256").update(fullText, "utf8").digest("hex");
    return { id, text: fullText, sha256 };
}
