/*
 * A newly learned date-difference Skill can be APPLIED before READY,
 * but only when a prior independently verified native date_diff trace
 * proves its execution program works.
 */
export function hasVerifiedDateSkillEvidence(skill) {
    if (!skill || skill.enabled === false) return false;
    const learning = skill.learning || {};
    const confidence = Number(learning.confidence ?? skill.confidence ?? 0);
    const successCount = Number(learning.successCount ?? 0);
    if (confidence < 0.70 || successCount < 1) return false;

    const examples = Array.isArray(skill.examples) ? skill.examples : [];
    return examples.some(example => {
        const result = example?.result;
        const validation = result?.validation;
        const tools = example?.executedTools
            || result?.executionMeta?.usedTools;
        return example.success === true
            && result?.success === true
            && result?.verified === true
            && result?.status === "COMPLETED"
            && validation?.valid === true
            && validation?.degraded !== true
            && Array.isArray(tools)
            && tools.includes("date_diff");
    });
}
