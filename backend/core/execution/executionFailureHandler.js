/*
 * =========================================================
 * JESSICA EXECUTION FAILURE HANDLER v5
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
 * Analyze
 *    ↓
 *
 * RETRY
 * REPLAN
 * CLARIFICATION
 * FINISH
 *
 *
 * Handler:
 *
 * - принимает решение;
 * - формирует следующий action.
 *
 *
 * НЕ:
 *
 * - выполняет retry;
 * - вызывает Planner;
 * - создаёт Plan;
 * - изменяет Context;
 * - сохраняет Learning.
 *
 * =========================================================
 */



import {
    shouldRetryExecution
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
 * FAILURE CATEGORIES
 * =========================================================
 */


const REPLAN_TYPES = new Set([


    "validation",

    "invalid-result",

    "strategy-failed",

    "wrong-route",

    "missing-data",

    "tool-mismatch",

    "planner-required"


]);









/*
 * =========================================================
 * NORMALIZE FAILURE
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

            String(
                failure.category ||
                detectCategory(
                    failure.failureType
                )
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
 * CATEGORY DETECTION
 * =========================================================
 */


function detectCategory(
    failureType
) {


    const type =

        String(
            failureType || ""
        )
        .toLowerCase();



    if (
        type.includes(
            "validation"
        )
    ) {

        return "validation";

    }



    if (
        type.includes(
            "tool"
        )
    ) {

        return "tool";

    }



    if (
        type.includes(
            "data"
        )
    ) {

        return "data";

    }



    if (
        type.includes(
            "planner"
        )
    ) {

        return "planner";

    }



    return "execution";

}









/*
 * =========================================================
 * DECISION BUILDER
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

            action === FAILURE_ACTION.RETRY ||

            action === FAILURE_ACTION.REPLAN,



        failure,



        metadata:

        {


            attempt:

                Number(
                    context?.attempt || 0
                ),



            stage:

                failure.stage,



            category:

                failure.category



        }

    };

}









/*
 * =========================================================
 * SHOULD REPLAN
 * =========================================================
 */


function shouldReplan(
    failure
) {


    return (

        REPLAN_TYPES.has(
            failure.failureType
        )

        ||

        REPLAN_TYPES.has(
            failure.category
        )

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


    /*
     * =====================================================
     * ANALYZE
     * =====================================================
     */


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
     * USER CLARIFICATION
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

        shouldRetryExecution(

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

        shouldReplan(
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
