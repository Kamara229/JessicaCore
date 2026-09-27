/*
 * =========================================================
 * JESSICA EXECUTION FAILURE HANDLER v8
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
 * Ответственность:
 *
 * - классифицировать ошибку;
 * - выбрать следующий action.
 *
 *
 * НЕ:
 *
 * - выполняет Retry;
 * - создаёт Plan;
 * - вызывает Planner;
 * - изменяет Context;
 * - сохраняет Learning.
 *
 * =========================================================
 */


import {
    shouldRetryExecution
} from "./retryPolicy.js";


import {
    canReplan
} from "./replanCoordinator.js";









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



            failureType:

                "unknown",



            category:

                "execution",



            reason:

                "Неизвестная ошибка",



            needsClarification:

                false,



            noVerifiedResult:

                false


        };

    }








    return {


        stage:

            failure.stage ||

            "execution",



        failureType:

            failure.failureType ||

            "execution-error",



        category:

            detectCategory(
                failure
            ),



        reason:

            failure.reason ||

            "Ошибка выполнения",



        validation:

            failure.validation ||
            null,



        needsClarification:

            failure.needsClarification === true,



        noVerifiedResult:

            failure.noVerifiedResult === true


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

        type.includes("timeout")

        ||

        type.includes("network")

        ||

        type.includes("temporary")

    ) {

        return "temporary";

    }







    if (

        type.includes("planner")

        ||

        type.includes("strategy")

    ) {

        return "planner";

    }







    if (

        type.includes("data")

    ) {

        return "data";

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


        type:

            action,



        reason:

            failure.reason,



        failure,



        canContinue:

            action === FAILURE_ACTION.RETRY

            ||

            action === FAILURE_ACTION.REPLAN,



        metadata:

        {


            executionId:

                context?.executionId ||
                null,



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



            failureType:

                failure.failureType,



            category:

                failure.category


        }


    };


}









/*
 * =========================================================
 * RETRY CHECK
 * =========================================================
 */


function canRetry(

    context,

    failure

) {


    return shouldRetryExecution(

        context,

        failure

    );


}









/*
 * =========================================================
 * MAIN HANDLER
 * =========================================================
 */


export async function handleExecutionFailure(

    context,

    failure

) {


    const normalized =

        normalizeFailure(
            failure
        );









    /*
     * =====================================================
     * CLARIFICATION
     * =====================================================
     */


    if (

        normalized.needsClarification === true

    ) {


        return buildDecision(

            FAILURE_ACTION.CLARIFICATION,

            normalized,

            context

        );

    }









    /*
     * =====================================================
     * NO VERIFIED RESULT
     * =====================================================
     */


    if (

        normalized.noVerifiedResult === true

    ) {


        return buildDecision(

            FAILURE_ACTION.FINISH,

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

            context

        )

        &&

        normalized.category !== "temporary"

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









export function isClarificationAction(

    decision

) {


    return (

        decision?.action ===

        FAILURE_ACTION.CLARIFICATION

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
