/**
 * Deterministic safeguards for claims based only on user-provided documents.
 * No external regulations, prices or certifications are independently checked.
 */

/** Avoid substring matching: "5000" must not match "15000". */
export function numericTokens(text) {
    const normalized = String(text || "").normalize("NFKC")
        .replace(/[\u00A0\u202F]/g, " ")
        .replace(/[−‐‑‒–—]/g, "-");
    return [...normalized.matchAll(/(?<![\p{L}\p{N}])\d+(?:[.,]\d+)*(?![\p{L}\p{N}])/gu)]
        .map(m => m[0].replace(/,/g, "."));
}

export function evidenceSupportsClaimNumbers(statement, evidence) {
    const sourceNumbers = new Set(
        evidence.flatMap(x => numericTokens(x.sourceText))
    );
    return numericTokens(statement).every(value => sourceNumbers.has(value));
}

/**
 * Attribution rule: legal/safety claims found in a *private document* are
 * the author's position, not an independently established regulation.
 */
export function attributedKind(text, kind) {
    if (kind !== "source_fact") return kind;
    const normative = /(?:не\s+)?(?:соответств\p{L}*|запрещ\p{L}*|разреш\p{L}*|допуска\p{L}*|обязательн\p{L}*|наруша\p{L}*|незаконн\p{L}*|правомерн\p{L}*|огнестойк\p{L}*|противопожарн\p{L}*|безопасн\p{L}*|требован\p{L}*|норматив\p{L}*)/iu;
    const regulatoryCitation = /(?:^|\s)(?:СП|ГОСТ|ФЗ|СНиП)\s*(?:№\s*)?\d/iu;
    return normative.test(text) || regulatoryCitation.test(text)
        ? "author_claim"
        : kind;
}
