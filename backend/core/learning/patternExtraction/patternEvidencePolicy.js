export function hasComparisonSourceCitations(taskInput, answerInput) {
    const task = String(taskInput || "");
    const answer = String(answerInput || "");
    if (!/(?:сравн\p{L}*|сопостав\p{L}*|compare|comparison)/iu.test(task)) {
        return true;
    }
    const urls = [...new Set((task.match(/https?:\/\/[^\s<>"')\]]+/gi) || [])
        .map(url => url.replace(/[.,;!?]+$/g,"").toLowerCase()))];
    if (urls.length !== 2) return true;
    // Old trace formats contain no per-claim provenance. Do not promote
    // an apparent comparison without at least explicit source attribution.
    return urls.every(url => answer.toLowerCase().includes(url));
}

/*
 * Reject unverified or incomplete autonomous discovery records.
 * A complex-task summary marked COMPLETED is not sufficient evidence.
 */
export function hasVerifiedPatternEvidence(evidence) {
    return hasComparisonSourceCitations(evidence?.task, evidence?.answer)
        && evidence?.result?.success === true
        && evidence?.result?.verified === true
        && evidence?.validation?.valid === true
        && evidence?.validation?.degraded !== true
        && String(evidence?.result?.status || "").toUpperCase() === "COMPLETED"
        && typeof evidence?.answer === "string"
        && evidence.answer.trim().length > 0;
}
