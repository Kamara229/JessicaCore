/*
 * =========================================================
 * JESSICA PLAN NORMALIZER
 * =========================================================
 *
 * Приводит сырой ответ Planner AI
 * к стабильному формату.
 *
 *
 * Ответственность:
 *
 * AI JSON
 *    ↓
 * normalize
 *    ↓
 * validate
 *
 *
 * Этот модуль НЕ:
 *
 * - проверяет существование tools;
 * - проверяет ссылки;
 * - проверяет evidence;
 * - принимает решения.
 *
 * =========================================================
 */



const EVIDENCE_MODES =
    new Set([
        "none",
        "search_results",
        "source_content"
    ]);



const MAX_STEPS =
    15;



/*
 * =========================================================
 * SAFE STRING
 * =========================================================
 */


function safeString(
    value
) {

    return typeof value === "string"
        ? value.trim()
        : "";

}



/*
 * =========================================================
 * NORMALIZE BOOLEAN
 * =========================================================
 */


function normalizeBoolean(
    value
) {

    if (
        value === true ||
        value === "true"
    ) {

        return true;

    }


    if (
        value === false ||
        value === "false"
    ) {

        return false;

    }


    return false;

}



/*
 * =========================================================
 * NORMALIZE EVIDENCE
 * =========================================================
 */


function normalizeEvidence(
    evidence
) {


    const mode =
        EVIDENCE_MODES.has(
            evidence?.mode
        )
            ? evidence.mode
            : "none";


    return {

        mode,


        reason:
            safeString(
                evidence?.reason
            )
                .slice(
                    0,
                    1000
                )

    };

}



/*
 * =========================================================
 * NORMALIZE ARGUMENTS
 * =========================================================
 */


function normalizeArguments(
    value
) {


    if (
        !value ||
        typeof value !== "object" ||
        Array.isArray(
            value
        )
    ) {

        return {};

    }


    return {
        ...value
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
            safeString(
                step.id
            )
            ||
            `step_${index + 1}`,


        tool:
            safeString(
                step.tool
            ),


        arguments:
            normalizeArguments(
                step.arguments
            )

    };

}



/*
 * =========================================================
 * NORMALIZE STEPS
 * =========================================================
 */


function normalizeSteps(
    steps
) {


    if (
        !Array.isArray(
            steps
        )
    ) {

        return [];

    }


    return steps

        .slice(
            0,
            MAX_STEPS
        )

        .map(
            normalizeStep
        )

        .filter(
            Boolean
        );

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
        normalizeBoolean(
            rawPlan.requiresTools
        );



    let steps =
        normalizeSteps(
            rawPlan.steps
        );



    /*
     * Если tools не нужны,
     * шагов быть не должно.
     */


    if (
        requiresTools === false
    ) {

        steps =
            [];

    }



    return {


        intent:
            safeString(
                rawPlan.intent
            )
            ||
            "unknown",



        requiresTools,



        reasoningSummary:
            safeString(
                rawPlan.reasoningSummary
            )
            .slice(
                0,
                2000
            ),



        evidence:
            normalizeEvidence(
                rawPlan.evidence
            ),



        steps


    };


}
