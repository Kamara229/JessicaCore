/*
 * =========================================================
 * JESSICA EXECUTION
 * FAILURE DECISION v3
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









function buildDecision(

    action,

    failure,

    context

){

    return {


        action,


        type:

            action,



        reason:

            failure.reason || "",



        failure,



        canContinue:

            action === FAILURE_ACTION.RETRY

            ||

            action === FAILURE_ACTION.REPLAN,



        metadata:

        {

            executionId:

                context?.executionId || null,


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

                failure.failureType || null,



            category:

                failure.category || null


        }


    };

}









function canRetry(

    context,

    failure

){

    return shouldRetryExecution(

        context,

        failure

    );

}









export function decideFailureAction(

    context,

    failure = {}

){

    /*
     * USER INPUT
     */

    if(

        failure.needsClarification === true

        ||

        failure.category === "clarification"

    ){

        return buildDecision(

            FAILURE_ACTION.CLARIFICATION,

            failure,

            context

        );

    }









    /*
     * NO VERIFIED RESULT
     */

    if(

        failure.noVerifiedResult === true

        ||

        failure.category === "no_verified"

    ){

        return buildDecision(

            FAILURE_ACTION.FINISH,

            failure,

            context

        );

    }









    /*
     * RETRY
     */

    if(

        canRetry(

            context,

            failure

        )

    ){

        return buildDecision(

            FAILURE_ACTION.RETRY,

            failure,

            context

        );

    }









    /*
     * REPLAN
     */

    if(

        canReplan(

            context

        )

        &&

        failure.category !== "temporary"

    ){

        return buildDecision(

            FAILURE_ACTION.REPLAN,

            failure,

            context

        );

    }









    /*
     * FINISH
     */

    return buildDecision(

        FAILURE_ACTION.FINISH,

        failure,

        context

    );


}
