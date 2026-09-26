/*
 * =========================================================
 * JESSICA PLAN NORMALIZER v3
 * =========================================================
 *
 * Приводит ответ Planner AI
 * к стабильному Execution Plan.
 *
 *
 * Flow:
 *
 * Planner AI JSON
 *        ↓
 * Normalize
 *        ↓
 * Validate
 *        ↓
 * Execution
 *
 *
 * Этот модуль НЕ:
 *
 * - проверяет tools;
 * - проверяет зависимости;
 * - проверяет корректность evidence;
 * - принимает решения.
 *
 * =========================================================
 */





const MAX_STEPS =
    15;


const MAX_TEXT_LENGTH =
    2000;








/*
 * =========================================================
 * SAFE HELPERS
 * =========================================================
 */


function safeString(
    value
) {

    return typeof value === "string"

        ? value.trim()

        : "";

}





function safeBoolean(
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





function safeArray(
    value
) {

    return Array.isArray(value)

        ? value

        : [];

}





function safeObject(
    value
) {


    if (
        !value ||
        typeof value !== "object" ||
        Array.isArray(value)
    ) {

        return {};

    }


    return value;

}








/*
 * =========================================================
 * EVIDENCE
 * =========================================================
 */


function normalizeEvidence(
    evidence
) {


    const source =
        safeObject(
            evidence
        );



    const allowedModes = [

        "none",

        "search_results",

        "source_content"

    ];



    return {


        mode:

            allowedModes.includes(
                source.mode
            )

                ? source.mode

                : "none",



        reason:

            safeString(
                source.reason
            )
            .slice(
                0,
                1000
            )


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


    const source =
        safeObject(
            experience
        );



    return {


        used:

            safeBoolean(
                source.used
            ),



        source:

            safeString(
                source.source
            )
            ||
            null,



        skills:

            safeArray(
                source.skills
            )
            .map(

                skill => {

                    if (
                        typeof skill === "string"
                    ) {

                        return skill.trim();

                    }


                    if (
                        skill &&
                        typeof skill === "object"
                    ) {

                        return {

                            id:
                                skill.id ||
                                null,


                            name:
                                skill.name ||
                                ""

                        };

                    }


                    return null;

                }

            )
            .filter(Boolean)


    };


}









/*
 * =========================================================
 * ARGUMENTS
 * =========================================================
 */


function normalizeArguments(
    argumentsValue
) {


    return {

        ...safeObject(
            argumentsValue
        )

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
         * Служебные поля
         *
         * сохраняем для Execution Trace
         */


        experienceUsed:

            safeBoolean(
                step.experienceUsed
            )

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


    return safeArray(
        steps
    )

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
 * EXTRA METADATA
 * =========================================================
 *
 * Сохраняем полезные поля,
 * которые AI может добавить.
 *
 * Нужно для:
 *
 * - Learning;
 * - Analytics;
 * - Trace.
 *
 * =========================================================
 */


function normalizeMetadata(
    plan
) {


    return {


        plannerVersion:

            safeString(
                plan.plannerVersion
            )
            ||
            null,



        generatedAt:

            safeString(
                plan.generatedAt
            )
            ||
            null


    };


}









/*
 * =========================================================
 * MAIN NORMALIZER
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






    return {


        /*
         * Основные параметры
         */


        intent:

            safeString(
                rawPlan.intent
            )

            ||

            "unknown",





        requiresTools:

            safeBoolean(
                rawPlan.requiresTools
            ),





        reasoningSummary:

            safeString(
                rawPlan.reasoningSummary
            )
            .slice(
                0,
                MAX_TEXT_LENGTH
            ),





        /*
         * Evidence
         */


        evidence:

            normalizeEvidence(
                rawPlan.evidence
            ),





        /*
         * Experience
         */


        experience:

            normalizeExperience(
                rawPlan.experience
            ),





        /*
         * Execution Steps
         */


        steps:

            normalizeSteps(
                rawPlan.steps
            ),





        /*
         * Metadata
         */


        metadata:

            normalizeMetadata(
                rawPlan
            )


    };


}
