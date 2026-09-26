/*
 * =========================================================
 * JESSICA RETRY POLICY v5
 * =========================================================
 *
 * Политика повторных попыток Execution.
 *
 *
 * Отвечает только:
 *
 * - можно ли повторить текущий маршрут;
 * - сколько попыток осталось.
 *
 *
 * НЕ:
 *
 * - создаёт новый Plan;
 * - вызывает Planner;
 * - делает Replan;
 * - выполняет Tools;
 * - анализирует качество результата.
 *
 * =========================================================
 */







/*
 * =========================================================
 * CONFIG
 * =========================================================
 */


export const MAX_EXECUTION_ATTEMPTS = 3;









/*
 * =========================================================
 * RETRYABLE CATEGORIES
 * =========================================================
 *
 * Ошибки, где повтор того же плана
 * может дать другой результат.
 *
 * =========================================================
 */


const RETRYABLE_CATEGORIES = new Set([


    "temporary",


    "network",


    "timeout",


    "rate-limit",


    "service-unavailable",


    "tool-temporary"



]);









/*
 * =========================================================
 * RETRYABLE TYPES
 * =========================================================
 */


const RETRYABLE_TYPES = new Set([


    "temporary-error",


    "timeout-error",


    "network-error",


    "tool-error",


    "runner-error"



]);









/*
 * =========================================================
 * NEVER RETRY
 * =========================================================
 *
 * Эти ошибки требуют:
 *
 * - Replan;
 * - уточнение;
 * - завершение.
 *
 * =========================================================
 */


const NON_RETRYABLE_TYPES = new Set([


    "user-required",


    "needs-clarification",


    "validation-error",


    "validation-failure",


    "invalid-result",


    "missing-data",


    "wrong-tool",


    "planner-required"



]);









/*
 * =========================================================
 * NORMALIZE NUMBER
 * =========================================================
 */


function normalizeNumber(
    value
) {


    const number =
        Number(
            value
        );


    if (
        Number.isFinite(
            number
        )
    ) {

        return number;

    }


    return 0;

}









/*
 * =========================================================
 * SHOULD RETRY
 * =========================================================
 */


export function shouldRetryExecution(

    context,

    failure

) {


    /*
     * Лимит попыток
     */


    const attempt =

        normalizeNumber(
            context?.attempt
        );



    if (
        attempt >= MAX_EXECUTION_ATTEMPTS
    ) {

        return false;

    }







    /*
     * Нет ошибки
     */


    if (
        !failure ||
        typeof failure !== "object"
    ) {

        return false;

    }







    /*
     * Требуется пользователь
     */


    if (
        failure.needsClarification === true
    ) {

        return false;

    }








    const type =

        String(
            failure.failureType || ""
        )
        .toLowerCase();




    const category =

        String(
            failure.category || ""
        )
        .toLowerCase();









    /*
     * Ошибки, которые нельзя повторять.
     */


    if (

        NON_RETRYABLE_TYPES.has(
            type
        )

    ) {

        return false;

    }








    /*
     * Явно временная ошибка.
     */


    if (

        RETRYABLE_TYPES.has(
            type
        )

    ) {

        return true;

    }








    /*
     * Категория временной ошибки.
     */


    if (

        RETRYABLE_CATEGORIES.has(
            category
        )

    ) {

        return true;

    }








    /*
     * По умолчанию:
     *
     * не повторяем неизвестное.
     *
     * Неизвестная ошибка должна
     * уйти в Replan или Terminal.
     *
     */


    return false;

}









/*
 * =========================================================
 * SHOULD STOP
 * =========================================================
 */


export function shouldStopExecution(

    context

) {


    return (

        normalizeNumber(
            context?.attempt
        )

        >=

        MAX_EXECUTION_ATTEMPTS

    );

}









/*
 * =========================================================
 * REMAINING ATTEMPTS
 * =========================================================
 */


export function getRemainingAttempts(

    attempt

) {


    return Math.max(

        0,

        MAX_EXECUTION_ATTEMPTS -

        normalizeNumber(
            attempt
        )

    );

}









/*
 * =========================================================
 * CHECK FAILURE
 * =========================================================
 */


export function isRetryableFailure(

    failure

) {


    if (
        !failure
    ) {

        return false;

    }



    return (


        RETRYABLE_TYPES.has(

            String(
                failure.failureType || ""
            )
            .toLowerCase()

        )


        ||


        RETRYABLE_CATEGORIES.has(

            String(
                failure.category || ""
            )
            .toLowerCase()

        )


    );

}
