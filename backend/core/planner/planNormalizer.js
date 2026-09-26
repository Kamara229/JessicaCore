/*
 * =========================================================
 * JESSICA PLAN NORMALIZER
 * =========================================================
 *
 * Приводит ответ Planner AI
 * к стабильному Execution Plan.
 *
 *
 * AI JSON
 *      ↓
 * Normalize
 *      ↓
 * Validate
 *
 *
 * НЕ:
 *
 * - проверяет tools;
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
 * BOOLEAN
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



    return false;

}








/*
 * =========================================================
 * EVIDENCE
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
 * ARGUMENTS
 * =========================================================
 */


function normalizeArguments(
    value
) {


    if (
        !value ||
        typeof value !== "object" ||
        Array.isArray(value)
    ) {

        return {};

    }



    return {

        ...value

    };

}








/*
 * =========================================================
 * EXPERIENCE
 * =========================================================
 */


function normalizeExperience(
    experience
) {


    if (
        !experience ||
        typeof experience !== "object"
    ) {

        return {


            used:
                false,


            skills:
                []

        };

    }




    return {


        used:

            experience.used === true,



        skills:

            Array.isArray(
                experience.skills
            )

                ? experience.skills

                    .map(

                        item =>

                            safeString(
                                item
                            )

                    )

                    .filter(Boolean)

                : []

    };

}








/*
 * =========================================================
 * STEP
 * =========================================================
 */


function normalizeStep(
    step,
    index
) {


    if (
        !step ||
        typeof step !== "object" ||
        Array.isArray(step)
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
            ),



        /*
         * Сохраняем дополнительные поля
         * для Execution Trace.
         */


        experienceUsed:

            step.experienceUsed === true



    };

}








/*
 * =========================================================
 * STEPS
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

        .filter(Boolean);

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
        Array.isArray(rawPlan)
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
     * Если инструменты не нужны,
     * выполнение не требуется.
     */


    if (
        !requiresTools
    ) {

        steps = [];

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





        /*
         * Связь:
         *
         * Plan
         *  ↓
         * Experience
         *  ↓
         * Learning
         */


        experienceUsed:

            rawPlan.experienceUsed === true,



        experience:

            normalizeExperience(
                rawPlan.experience
            ),





        evidence:

            normalizeEvidence(
                rawPlan.evidence
            ),





        steps



    };

}
