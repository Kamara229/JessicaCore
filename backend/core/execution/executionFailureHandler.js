/*
 * =========================================================
 * JESSICA EXECUTION FAILURE HANDLER v4
 * =========================================================
 *
 * Центральный маршрутизатор ошибок Execution.
 *
 *
 * Flow:
 *
 * Execution Failure
 *        ↓
 * Normalize
 *        ↓
 * Analyze
 *        ↓
 *
 * RETRY
 * REPLAN
 * CLARIFICATION
 * FINISH
 *
 *
 * Handler только принимает решение.
 *
 *
 * НЕ:
 *
 * - выполняет retry;
 * - создаёт новый Plan;
 * - вызывает Planner;
 * - создаёт Answer;
 * - сохраняет Learning;
 * - меняет Experience.
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
 * FAILURE ACTIONS
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


            reason:
                "Неизвестная ошибка",


            validation:
                null

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
 * BUILD DECISION
 * =========================================================
 */


function createDecision(

    action,

    failure,

    context

) {


    return {


        action,


        reason:

            failure.reason,



        failure,



        metadata:
        {

            attempt:

                Number(
                    context?.attempt || 0
                ),



            stage:

                failure.stage,



            failureType:

                failure.failureType

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


    const replannableErrors =

        new Set([

            "validation-error",

            "invalid-result",

            "wrong-tool",

            "missing-data",

            "planner-required",

            "execution-strategy-failed"

        ]);



    return replannableErrors.has(

        failure.failureType

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
     * CLARIFICATION
     * =====================================================
     */


    if (

        analyzed?.needsClarification === true

    ) {


        return createDecision(

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


        return createDecision(

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


        return createDecision(

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


    return createDecision(

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
