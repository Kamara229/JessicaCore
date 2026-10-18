/*
 * Reject unverified or incomplete autonomous discovery records.
 * A complex-task summary marked COMPLETED is not sufficient evidence.
 */
export function hasVerifiedPatternEvidence(evidence) {
    return evidence?.result?.success === true
        && evidence?.result?.verified === true
        && evidence?.validation?.valid === true
        && evidence?.validation?.degraded !== true
        && String(evidence?.result?.status || "").toUpperCase() === "COMPLETED"
        && typeof evidence?.answer === "string"
        && evidence.answer.trim().length > 0;
}
