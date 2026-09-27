/*
 * =========================================================
 * JESSICA EXECUTION
 * FAILURE DECISION v1
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

            action === "RETRY"

            ||

            action === "REPLAN",



        metadata:

        {


            executionId:

                context?.executionId ||

                null,



            attempt:

                Number(

                    context?.attempt ||

                    0

                ),



            retryCount:

                Number(

                    context?.retryCount ||

                    0

                ),



            replanCount:

                Number(

                    context?.replanCount ||

                    0

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
 * RETRY
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
 * DECIDE ACTION
 * =========================================================
 */


export function decideFailureAction(

    context,

    failure

) {





    /*
     * =====================================================
     * USER CLARIFICATION
     * =====================================================
     */


    if(

        failure.needsClarification === true

    ) {


        return buildDecision(

            "CLARIFICATION",

            failure,

            context

        );

    }









    /*
     * =====================================================
     * NO VERIFIED RESULT
     * =====================================================
     */


    if(

        failure.noVerifiedResult === true

    ) {


        return buildDecision(

            "FINISH",

            failure,

            context

        );

    }









    /*
     * =====================================================
     * RETRY
     * =====================================================
     */


    if(

        canRetry(

            context,

            failure

        )

    ) {


        return buildDecision(

            "RETRY",

            failure,

            context

        );

    }









    /*
     * =====================================================
     * REPLAN
     * =====================================================
     */


    if(

        canReplan(

            context

        )

        &&

        failure.category !== "temporary"

    ) {


        return buildDecision(

            "REPLAN",

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

        "FINISH",

        failure,

        context

    );


}
