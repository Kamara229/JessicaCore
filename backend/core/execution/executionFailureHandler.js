/*
 * =========================================================
 * JESSICA EXECUTION FAILURE HANDLER v11
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
 * Decision
 *
 *
 * НЕ:
 *
 * - выполняет Retry;
 * - выполняет Replan;
 * - изменяет Context;
 * - создаёт Plan;
 * - сохраняет Learning.
 *
 * =========================================================
 */


import {
    normalizeFailure
} from "./failure/failureNormalizer.js";


import {
    classifyFailure
} from "./failure/failureClassifier.js";


import {
    decideFailureAction
} from "./failure/failureDecision.js";


import {
    FAILURE_ACTION
} from "./failure/failureActions.js";


export {
    FAILURE_ACTION
};









/*
 * =========================================================
 * HANDLE FAILURE
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









    normalized.original =

        failure || null;









    /*
     * CLASSIFICATION
     */


    try {


        normalized.category =

            classifyFailure(

                normalized

            );


    }

    catch(error){


        normalized.category =

            "execution";


    }









    /*
     * NO CONTEXT
     */


    if(

        !context

    ){


        return {


            action:

                FAILURE_ACTION.FINISH,


            type:

                FAILURE_ACTION.FINISH,


            reason:

                "Execution context отсутствует",


            failure:

                normalized


        };


    }









    /*
     * DECISION
     */


    return decideFailureAction(

        context,

        normalized

    );


}









/*
 * =========================================================
 * HELPERS
 * =========================================================
 */


export function isRetryAction(

    decision

){


    return (

        decision?.action ===

        FAILURE_ACTION.RETRY

    );


}









export function isReplanAction(

    decision

){


    return (

        decision?.action ===

        FAILURE_ACTION.REPLAN

    );


}









export function isClarificationAction(

    decision

){


    return (

        decision?.action ===

        FAILURE_ACTION.CLARIFICATION

    );


}









export function isTerminalAction(

    decision

){


    return (

        decision?.action ===

        FAILURE_ACTION.FINISH

    );


}
