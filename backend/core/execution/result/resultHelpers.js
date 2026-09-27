/*
 * =========================================================
 * JESSICA RESULT HELPERS v2
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
 * VALID RESULT
 * =========================================================
 */


function hasStatus(

    result,

    status

) {


    return (

        result &&

        typeof result === "object" &&

        result.status === status

    );

}









/*
 * =========================================================
 * COMPLETED
 * =========================================================
 */


export function isCompletedResult(

    result

) {


    return (

        hasStatus(

            result,

            EXECUTION_RESULT_STATUS.COMPLETED

        )

        &&

        result.success === true

        &&

        result.verified === true

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

        hasStatus(

            result,

            EXECUTION_RESULT_STATUS.FAILED

        )

        &&

        result.success === false

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

        hasStatus(

            result,

            EXECUTION_RESULT_STATUS
                .NEEDS_CLARIFICATION

        )

        &&

        result.success === false

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

        hasStatus(

            result,

            EXECUTION_RESULT_STATUS
                .NO_VERIFIED_RESULT

        )

        &&

        result.success === false

    );

}
