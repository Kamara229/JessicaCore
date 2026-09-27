/*
 * =========================================================
 * JESSICA EXECUTION
 * FAILURE DECISION v2
 * =========================================================
 *
 * Выбор следующего Execution Action.
 *
 *
 * Flow:
 *
 * Failure
 *      ↓
 * Decision
 *      ↓
 *
 * RETRY
 * REPLAN
 * CLARIFICATION
 * FINISH
 *
 *
 * НЕ:
 *
 * - выполняет действие;
 * - меняет Context;
 * - запускает Execution.
 *
 * =========================================================
 */


import {
    shouldRetryExecution
} from "../retryPolicy.js";


import {
    canReplan
} from "../replanCoordinator.js";


import {
    FAILURE_ACTION
} from "./failureActions.js";









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

                failure.category || null


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


    /*
     * Явный сигнал от слоя ниже
     */


    if(

        failure.shouldRetry === true

    ){

        return true;

    }









    return shouldRetryExecution(

        context,

        failure

    );

}









/*
 * =========================================================
 * DECIDE ACTION
 * =========================================================
 */


export function decideFailureAction(

    context,

    failure

) {


    /*
     * =====================================================
     * USER INPUT REQUIRED
     * =====================================================
     */


    if (

        failure.needsClarification === true

        ||

        failure.category === "clarification"

    ) {


        return buildDecision(

            FAILURE_ACTION.CLARIFICATION,

            failure,

            context

        );

    }









    /*
     * =====================================================
     * RESULT NOT VERIFIED
     * =====================================================
     */


    if (

        failure.noVerifiedResult === true

        ||

        failure.category === "no_verified"

    ) {


        return buildDecision(

            FAILURE_ACTION.FINISH,

            failure,

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

            failure

        )

    ) {


        return buildDecision(

            FAILURE_ACTION.RETRY,

            failure,

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

        failure.category !== "temporary"

    ) {


        return buildDecision(

            FAILURE_ACTION.REPLAN,

            failure,

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

        failure,

        context

    );


}
