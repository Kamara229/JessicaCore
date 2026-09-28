/*
 * =========================================================
 * JESSICA PLANNER ATTEMPT
 * =========================================================
 *
 * Выполняет ОДНУ попытку построения плана.
 *
 *
 * Flow:
 *
 * Planner Request
 *      ↓
 * Parse
 *      ↓
 * Normalize
 *      ↓
 * Validate
 *      ↓
 * Result
 *
 *
 * НЕ:
 *
 * - управляет retry loop;
 * - делает sleep;
 * - классифицирует network errors;
 * - решает количество попыток.
 *
 * =========================================================
 */


import {
    requestPlan
} from "../plannerRequest.js";


import {
    parsePlan
} from "../planParser.js";


import {
    normalizePlan
} from "../planNormalizer.js";


import {
    validatePlan
} from "../planValidator.js";


/*
 * =========================================================
 * ATTEMPT
 * =========================================================
 */


export async function runPlannerAttempt({

    task,

    feedback = "",

    context,

    attempt

}) {


    /*
     * =====================================================
     * REQUEST
     * =====================================================
     */


    const rawResponse =
        await requestPlan(

            task,

            feedback,

            context

        );


    /*
     * =====================================================
     * PARSE
     * =====================================================
     */


    const parsedPlan =
        parsePlan(
            rawResponse
        );


    if (
        !parsedPlan
    ) {


        throw new Error(
            "planner_parse_failed"
        );

    }


    /*
     * =====================================================
     * NORMALIZE
     * =====================================================
     */


    const normalizedPlan =
        normalizePlan(
            parsedPlan
        );


    if (
        !normalizedPlan
    ) {


        throw new Error(
            "planner_normalize_failed"
        );

    }


    /*
     * =====================================================
     * VALIDATE
     * =====================================================
     */


    const validation =
        validatePlan(

            normalizedPlan,

            context

        );


    if (
        !validation.success
    ) {


        return {

            success:
                false,

            type:
                "semantic",

            stage:
                "validation",

            feedback:

                validation.text ||

                "planner_validation_failed",

            plan:
                null,

            attempt

        };

    }


    /*
     * =====================================================
     * SUCCESS
     * =====================================================
     */


    return {

        success:
            true,

        type:
            "success",

        stage:
            "completed",

        feedback:
            "",

        plan:
            normalizedPlan,

        attempt

    };

}
