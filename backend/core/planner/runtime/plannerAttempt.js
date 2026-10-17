/*
 * =========================================================
 * JESSICA PLANNER ATTEMPT v3
 * =========================================================
 *
 * Planner Request
 *      ↓
 * Parse
 *      ↓
 * Normalize
 *      ↓
 * Validate
 *      ↓
 *
 * success
 *      → Plan
 *
 * semantic failure
 *      → Validation feedback
 *      → Rejected Plan
 *
 *
 * Rejected Plan нужен только
 * следующему Planner attempt
 * для построения repair feedback.
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


    if(
        !parsedPlan
    ){

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


    if(
        !normalizedPlan
    ){

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

            context,

            task

        );


    /*
     * =====================================================
     * SEMANTIC FAILURE
     * =====================================================
     */


    if(
        !validation.success
    ){

        return {

            success:
                false,

            type:
                "semantic",

            stage:
                "validation",

            feedback:

                validation.text
                ||
                "planner_validation_failed",


            /*
             * КРИТИЧЕСКО:
             *
             * Plan не выполняется.
             *
             * Он передаётся только
             * Feedback Builder'у,
             * чтобы следующий attempt
             * понимал, что именно
             * нельзя повторять.
             */


            rejectedPlan:
                normalizedPlan,

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

        rejectedPlan:
            null,

        plan:
            normalizedPlan,

        attempt

    };

}
