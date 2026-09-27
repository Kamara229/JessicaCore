/*
 * =========================================================
 * JESSICA RESULT HELPERS v3
 * =========================================================
 *
 * Проверки Execution Result.
 *
 *
 * Отвечает:
 *
 * - определение типа результата;
 * - безопасная проверка Result контрактов.
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
 * BASE CHECK
 * =========================================================
 */


function isObject(

    value

){

    return (

        value &&

        typeof value === "object"

    );

}









function hasStatus(

    result,

    status

){

    return (

        isObject(result)

        &&

        result.status === status

    );

}









/*
 * =========================================================
 * EXECUTION RESULT
 * =========================================================
 */


export function isExecutionResult(

    result

){

    return isObject(result)

        &&

        typeof result.status === "string";

}









/*
 * =========================================================
 * SUCCESS
 * =========================================================
 */


export function isSuccessfulResult(

    result

){

    return (

        isCompletedResult(result)

    );

}









/*
 * =========================================================
 * COMPLETED
 * =========================================================
 */


export function isCompletedResult(

    result

){


    return (


        hasStatus(

            result,

            EXECUTION_RESULT_STATUS.COMPLETED

        )

        &&


        result.success === true

        &&


        result.verified === true

        &&


        Boolean(

            result.answer

        )


    );

}









/*
 * =========================================================
 * FAILED
 * =========================================================
 */


export function isFailedResult(

    result

){


    return (


        hasStatus(

            result,

            EXECUTION_RESULT_STATUS.FAILED

        )

        &&


        result.success === false

        &&


        Boolean(

            result.failure

        )


    );

}









/*
 * =========================================================
 * CLARIFICATION
 * =========================================================
 */


export function isClarificationResult(

    result

){


    return (


        hasStatus(

            result,

            EXECUTION_RESULT_STATUS
                .NEEDS_CLARIFICATION

        )

        &&


        result.success === false

        &&


        Boolean(

            result.clarification

        )


    );

}









/*
 * =========================================================
 * NO VERIFIED
 * =========================================================
 */


export function isNoVerifiedResult(

    result

){


    return (


        hasStatus(

            result,

            EXECUTION_RESULT_STATUS
                .NO_VERIFIED_RESULT

        )

        &&


        result.success === false

        &&


        result.verified === false


    );

}









/*
 * =========================================================
 * FAILURE CHECK
 * =========================================================
 */


export function hasFailure(

    result

){

    return Boolean(

        result?.failure

    );

}









/*
 * =========================================================
 * TERMINAL CHECK
 * =========================================================
 */


export function hasTerminal(

    result

){

    return Boolean(

        result?.terminal

    );

}
