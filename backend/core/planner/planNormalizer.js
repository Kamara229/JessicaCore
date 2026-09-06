/*
 * =========================================================
 * JESSICA PLAN NORMALIZER
 * =========================================================
 *
 * Приводит сырой ответ Planner AI
 * к стабильной структуре.
 *
 * Этот модуль ничего не валидирует глубоко.
 * Он только нормализует данные.
 */


const EVIDENCE_MODES =
    new Set([
        "none",
        "search_results",
        "source_content"
    ]);


/*
 * =========================================================
 * NORMALIZE EVIDENCE
 * =========================================================
 */


function normalizeEvidence(
    rawEvidence
) {

    const mode =
        EVIDENCE_MODES.has(
            rawEvidence?.mode
        )
            ? rawEvidence.mode
            : "none";


    return {
        mode,

        reason:
            typeof rawEvidence?.reason === "string"
                ? rawEvidence.reason.trim()
                : ""
    };

}


/*
 * =========================================================
 * NORMALIZE STEP
 * =========================================================
 */


function normalizeStep(
    step,
    index
) {

    if (
        !step ||
        typeof step !== "object" ||
        Array.isArray(
            step
        )
    ) {

        return null;

    }


    return {
        id:
            typeof step.id === "string" &&
            step.id.trim()
                ? step.id.trim()
                : `step_${index + 1}`,

        tool:
            typeof step.tool === "string"
                ? step.tool.trim()
                : "",

        arguments:
            step.arguments &&
            typeof step.arguments === "object" &&
            !Array.isArray(
                step.arguments
            )
                ? step.arguments
                : {}
    };

}


/*
 * =========================================================
 * NORMALIZE PLAN
 * =========================================================
 */


export function normalizePlan(
    rawPlan
) {

    if (
        !rawPlan ||
        typeof rawPlan !== "object" ||
        Array.isArray(
            rawPlan
        )
    ) {

        return null;

    }


    const requiresTools =
        rawPlan.requiresTools;


    let steps =
        Array.isArray(
            rawPlan.steps
        )
            ? rawPlan.steps
                .map(
                    normalizeStep
                )
                .filter(
                    Boolean
                )
            : [];


    /*
     * Для задачи без tools
     * steps всегда пустой.
     */
    if (
        requiresTools === false
    ) {

        steps =
            [];

    }


    return {
        intent:
            typeof rawPlan.intent === "string"
                ? rawPlan.intent.trim()
                : "",

        requiresTools,

        reasoningSummary:
            typeof rawPlan.reasoningSummary === "string"
                ? rawPlan.reasoningSummary.trim()
                : "",

        evidence:
            normalizeEvidence(
                rawPlan.evidence
            ),

        steps
    };

}
