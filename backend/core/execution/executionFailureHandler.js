/*
 * =========================================================
 * JESSICA EXECUTION FAILURE HANDLER v3
 * =========================================================
 *
 * Координатор обработки ошибок выполнения.
 *
 *
 * Flow:
 *
 * Execution Failure
 *        ↓
 * Analyze Failure
 *        ↓
 *
 * RETRY
 * REPLAN
 * CLARIFICATION
 * FINISH
 *
 *
 * НЕ:
 *
 * - создаёт новый Plan;
 * - вызывает Planner;
 * - выполняет Tools;
 * - создаёт Answer;
 * - сохраняет Learning;
 * - изменяет Experience.
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


            reason:
                "Неизвестная ошибка",


            failureType:
                "unknown"


        };

    }





    return {


        stage:

            failure.stage ||
            "execution",



        reason:

            failure.reason ||
            "Ошибка выполнения",



        failureType:

            failure.failureType ||
            "execution-error",



        validation:

            failure.validation ||
            null



    };


}









/*
 * =========================================================
 * BUILD RETRY DATA
 * =========================================================
 */


function buildRetryDecision(
    context,
    failure
) {


    return {


        finished:
            false,


        action:
            FAILURE_ACTION.RETRY,


        retryContext:
        {

            attempt:
                context.attempt,


            failureType:
                failure.failureType,


            reason:
                failure.reason


        }



    };

}









/*
 * =========================================================
 * BUILD REPLAN DATA
 * =========================================================
 */


function buildReplanDecision(
    context,
    failure
) {


    return {


        finished:
            false,


        action:
            FAILURE_ACTION.REPLAN,


        replanContext:
        {

            task:
                context.task,


            previousPlan:
                context.plan,


            failure:
            {

                stage:
                    failure.stage,


                type:
                    failure.failureType,


                reason:
                    failure.reason


            },



            attempt:
                context.attempt



        }



    };

}









/*
 * =========================================================
 * BUILD CLARIFICATION
 * =========================================================
 */


function buildClarificationDecision(
    failure
) {


    return {


        finished:
            true,


        action:
            FAILURE_ACTION.CLARIFICATION,


        clarification:
        {


            reason:
                failure.reason,


            stage:
                failure.stage



        }


    };

}









/*
 * =========================================================
 * BUILD FINISH
 * =========================================================
 */


function buildFinishDecision(
    failure
) {


    return {


        finished:
            true,


        action:
            FAILURE_ACTION.FINISH,


        failure



    };

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


        return buildClarificationDecision(

            normalized

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


        return buildRetryDecision(

            context,

            normalized

        );


    }









    /*
     * =====================================================
     * REPLAN
     * =====================================================
     *
     * Ошибка передаётся Planner.
     *
     * Handler не создаёт план.
     *
     * =====================================================
     */


    if (

        [

            "validation-error",

            "invalid-result",

            "wrong-tool",

            "missing-data",

            "planner-required"

        ]

        .includes(
            normalized.failureType
        )

    ) {


        return buildReplanDecision(

            context,

            normalized

        );


    }









    /*
     * =====================================================
     * FINISH
     * =====================================================
     */


    return buildFinishDecision(

        normalized

    );


}
