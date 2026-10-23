/**
 * Fail-closed evidence guard for autonomous positive Skill reinforcement.
 * A completed step counter is not proof of a verified task result.
 * Keep this policy independent of queue/storage implementation.
 */
const NEGATIVE_STATUSES = new Set([
    "NO_VERIFIED_RESULT", "NEEDS_CLARIFICATION", "FAILED",
    "PARTIAL", "CANCELLED", "ERROR"
]);
const NEGATIVE_OUTCOMES = new Set([
    "no_verified_result", "needs_clarification", "validation_failed",
    "execution_failed", "failed"
]);
function recordBlocks(record) {
    if (!record || typeof record !== "object" || Array.isArray(record)) return false;
    const status = String(record.status ?? "").trim().toUpperCase();
    const outcome = String(record.outcomeType ?? record.outcome_type ?? "").trim().toLowerCase();
    return NEGATIVE_STATUSES.has(status) ||
        NEGATIVE_OUTCOMES.has(outcome) ||
        record.success === false ||
        record.degraded === true ||
        record.verified === false ||
        record.valid === false ||
        record.needsClarification === true;
}
/**
 * Explicitly negative verdicts override optimistic completion counters.
 * This does not reject missing evidence here; existing terminal/child
 * validation checks are responsible for positive evidence requirements.
 */
export function hasNegativeLearningEvidence(trace) {
    if (!trace || typeof trace !== "object" || Array.isArray(trace)) return true;
    const result = trace.result;
    const records = [trace, trace.validation, result, result?.validation];
    if (Array.isArray(result?.results))
        for (const child of result.results) records.push(child, child?.validation);
    return records.some(recordBlocks);
}
