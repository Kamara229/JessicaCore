/*
 * =========================================================
 * JESSICA RESULT HELPERS v1
 * =========================================================
 *
 * Проверки Execution Result.
 *
 *
 * Отвечает:
 *
 * - определение типа результата.
 *
 *
 * НЕ:
 *
 * - создаёт Result;
 * - изменяет Result;
 * - анализирует ошибки.
 *
 * =========================================================
 */


import {
    EXECUTION_RESULT_STATUS
} from "./resultStatus.js";









/*
 * =========================================================
 * COMPLETED
 * =========================================================
 */


export function isCompletedResult(

    result

) {


    return (

        result?.status ===

            EXECUTION_RESULT_STATUS.COMPLETED

        &&

        result?.success === true

    );

}









/*
 * =========================================================
 * FAILED
 * =========================================================
 */


export function isFailedResult(

    result

) {


    return (

        result?.status ===

            EXECUTION_RESULT_STATUS.FAILED

    );

}









/*
 * =========================================================
 * NEEDS CLARIFICATION
 * =========================================================
 */


export function isClarificationResult(

    result

) {


    return (

        result?.status ===

            EXECUTION_RESULT_STATUS
                .NEEDS_CLARIFICATION

    );

}









/*
 * =========================================================
 * NO VERIFIED RESULT
 * =========================================================
 */


export function isNoVerifiedResult(

    result

) {


    return (

        result?.status ===

            EXECUTION_RESULT_STATUS
                .NO_VERIFIED_RESULT

    );

}
