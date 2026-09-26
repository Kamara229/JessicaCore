/*
 * =========================================================
 * JESSICA EXECUTION FAILURE HANDLER v6
 * =========================================================
 *
 * Центральный маршрутизатор Execution Failure.
 *
 *
 * Flow:
 *
 * Failure
 *    ↓
 * Normalize
 *    ↓
 * Classify
 *    ↓
 *
 * RETRY
 * REPLAN
 * CLARIFICATION
 * FINISH
 *
 *
 * НЕ:
 *
 * - выполняет retry;
 * - вызывает Planner;
 * - изменяет Context;
 * - сохраняет Learning.
 *
 * =========================================================
 */



import {
    shouldRetryExecution,
    MAX_EXECUTION_ATTEMPTS
} from "./retryPolicy.js";


import {
    analyzeRunFailure
} from "./runFailurePolicy.js";









/*
 * =========================================================
 * ACTIONS
 * =========================================================
 */


export const FAILURE_ACTION = {


    RETRY:
        "RETRY",


    REPLAN:
        "REPLAN",


    CLARIFICATION:
        "CLARIFICATION",


    FINISH:
        "FINISH"


};









/*
 * =========================================================
 * CONFIG
 * =========================================================
 */


const MAX_REPLAN_COUNT = 2;









/*
 * =========================================================
 * REPLAN FAILURE TYPES
 * =========================================================
 */


const REPLAN_FAILURE_TYPES = new Set([


    "validation-error",

    "invalid-result",

    "wrong-route",

    "strategy-failed",

    "planner-required",

    "tool-mismatch",

    "missing-data"


]);









/*
 * =========================================================
 * RETRY FAILURE TYPES
 * =========================================================
 */


const TEMPORARY_FAILURE_TYPES = new Set([


    "timeout",

    "network-error",

    "temporary-error",

    "tool-error",

    "runner-error"


]);









/*
 * =========================================================
 * NORMALIZE
 * =========================================================
 */


function normalizeFailure(

    failure

) {


    if (
        !failure ||
        typeof failure !== "object"
    ) {


        return {

            stage:
                "execution",

            failureType:
                "unknown",

            category:
                "execution",

            reason:
                "Неизвестная ошибка"

        };

    }





    return {


        stage:

            String(
                failure.stage ||
                "execution"
            ),



        failureType:

            String(
                failure.failureType ||
                "execution-error"
            ),



        category:

            detectCategory(
                failure
            ),



        reason:

            String(
                failure.reason ||
                "Ошибка выполнения"
            ),



        validation:

            failure.validation ||
            null



    };

}









/*
 * =========================================================
 * CATEGORY
 * =========================================================
 */


function detectCategory(

    failure

) {


    const type =

        String(
            failure?.failureType || ""
        )
        .toLowerCase();




    if (
        type.includes("validation")
    ) {

        return "validation";

    }



    if (
        type.includes("tool")
    ) {

        return "tool";

    }



    if (
        type.includes("network")
        ||
        type.includes("timeout")
    ) {

        return "temporary";

    }



    if (
        type.includes("data")
    ) {

        return "data";

    }



    if (
        type.includes("planner")
    ) {

        return "planner";

    }



    return "execution";

}









/*
 * =========================================================
 * BUILD DECISION
 * =========================================================
 */


function buildDecision(

    action,

    failure,

    context

) {


    return {


        action,


        canContinue:

            action === FAILURE_ACTION.RETRY

            ||

            action === FAILURE_ACTION.REPLAN,



        failure,



        metadata:

        {


            attempt:

                Number(
                    context?.attempt || 0
                ),



            retryCount:

                Number(
                    context?.retryCount || 0
                ),



            replanCount:

                Number(
                    context?.replanCount || 0
                ),



            stage:

                failure.stage,



            category:

                failure.category,



            failureType:

                failure.failureType



        }

    };

}









/*
 * =========================================================
 * CHECK REPLAN
 * =========================================================
 */


function canReplan(

    context,

    failure

) {


    const replans =

        Number(
            context?.replanCount || 0
        );



    if (
        replans >= MAX_REPLAN_COUNT
    ) {

        return false;

    }



    return (

        REPLAN_FAILURE_TYPES.has(
            failure.failureType
        )

    );

}









/*
 * =========================================================
 * CHECK RETRY
 * =========================================================
 */


function canRetry(

    context,

    failure

) {


    if (
        !TEMPORARY_FAILURE_TYPES.has(
            failure.failureType
        )
    ) {


        return false;

    }



    return shouldRetryExecution(

        context,

        failure

    );

}









/*
 * =========================================================
 * HANDLE FAILURE
 * =========================================================
 */


export async function handleExecutionFailure(

    context,

    failure

) {


    const analyzed =

        analyzeRunFailure(
            failure
        )
        ||
        failure;





    const normalized =

        normalizeFailure(
            analyzed
        );









    /*
     * =====================================================
     * CLARIFICATION
     * =====================================================
     */


    if (
        analyzed?.needsClarification === true
    ) {


        return buildDecision(

            FAILURE_ACTION.CLARIFICATION,

            normalized,

            context

        );

    }









    /*
     * =====================================================
     * RETRY
     * =====================================================
     */


    if (

        canRetry(

            context,

            normalized

        )

    ) {


        return buildDecision(

            FAILURE_ACTION.RETRY,

            normalized,

            context

        );

    }









    /*
     * =====================================================
     * REPLAN
     * =====================================================
     */


    if (

        canReplan(

            context,

            normalized

        )

    ) {


        return buildDecision(

            FAILURE_ACTION.REPLAN,

            normalized,

            context

        );

    }









    /*
     * =====================================================
     * FINISH
     * =====================================================
     */


    return buildDecision(

        FAILURE_ACTION.FINISH,

        normalized,

        context

    );


}









/*
 * =========================================================
 * HELPERS
 * =========================================================
 */


export function isRetryAction(
    decision
) {

    return (

        decision?.action ===
        FAILURE_ACTION.RETRY

    );

}




export function isReplanAction(
    decision
) {

    return (

        decision?.action ===
        FAILURE_ACTION.REPLAN

    );

}




export function isTerminalAction(
    decision
) {

    return (

        decision?.action ===
        FAILURE_ACTION.FINISH

    );

}




export function isClarificationAction(
    decision
) {

    return (

        decision?.action ===
        FAILURE_ACTION.CLARIFICATION

    );

}
