/*
 * =========================================================
 * JESSICA EXECUTION FAILURE HANDLER v9
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
 * - собрать обработку Failure;
 * - передать ошибку между модулями;
 * - вернуть Decision.
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
 * HANDLE FAILURE
 * =========================================================
 */


export async function handleExecutionFailure(

    context,

    failure

) {


    /*
     * =====================================================
     * NORMALIZE
     * =====================================================
     */


    const normalized =

        normalizeFailure(

            failure

        );









    /*
     * =====================================================
     * CLASSIFY
     * =====================================================
     */


    normalized.category =

        classifyFailure(

            normalized

        );









    /*
     * =====================================================
     * DECISION
     * =====================================================
     */


    return decideFailureAction(

        context,

        normalized

    );


}









/*
 * =========================================================
 * ACTION HELPERS
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
