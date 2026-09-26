/*
 * =========================================================
 * JESSICA RETRY POLICY v4
 * =========================================================
 *
 * Политика повторных попыток Execution.
 *
 *
 * Отвечает только:
 *
 * Можно ли повторить текущий план?
 *
 *
 * НЕ:
 *
 * - анализирует ошибки;
 * - создаёт новый план;
 * - вызывает Planner;
 * - выполняет инструменты;
 * - работает с Validation.
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
 * RETRYABLE FAILURE TYPES
 * =========================================================
 */


const RETRYABLE_FAILURE_TYPES = new Set([


    /*
     * Временная проблема.
     *
     * Например:
     *
     * - timeout;
     * - network;
     * - временная недоступность API.
     *
     */


    "temporary-error",




    /*
     * Ошибка инструмента.
     *
     * Может пройти при повторном запуске.
     */


    "tool-error",




    /*
     * Ошибка выполнения маршрута.
     */


    "runner-error"



]);









/*
 * =========================================================
 * NEVER RETRY
 * =========================================================
 */


const TERMINAL_FAILURE_TYPES = new Set([


    /*
     * Нужны данные пользователя.
     */


    "user-required",




    /*
     * План неверный.
     *
     * Нужен Replan.
     */


    "validation-error",




    /*
     * Ответ невозможно использовать.
     */


    "invalid-result",




    /*
     * Недостаточно входных данных.
     */


    "missing-data"



]);









/*
 * =========================================================
 * NORMALIZE ATTEMPT
 * =========================================================
 */


function normalizeAttempt(
    value
) {


    const attempt =
        Number(
            value
        );


    return Number.isInteger(
        attempt
    )

        ? attempt

        : 0;

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
     * Проверка лимита.
     */


    const attempt =

        normalizeAttempt(
            context?.attempt
        );



    if (
        attempt >= MAX_EXECUTION_ATTEMPTS
    ) {


        return false;

    }








    /*
     * Нет ошибки —
     * повторять нечего.
     */


    if (
        !failure ||
        typeof failure !== "object"
    ) {


        return false;

    }








    const type =

        String(
            failure.failureType || ""
        )
        .trim();








    /*
     * Терминальные ошибки.
     */


    if (
        TERMINAL_FAILURE_TYPES.has(
            type
        )
    ) {


        return false;

    }








    /*
     * Разрешённые временные ошибки.
     */


    if (
        RETRYABLE_FAILURE_TYPES.has(
            type
        )
    ) {


        return true;

    }








    return false;

}









/*
 * =========================================================
 * REMAINING ATTEMPTS
 * =========================================================
 */


export function getRemainingAttempts(

    attempt

) {


    const current =

        normalizeAttempt(
            attempt
        );



    return Math.max(

        0,

        MAX_EXECUTION_ATTEMPTS - current

    );

}









/*
 * =========================================================
 * EXPORT CHECK
 * =========================================================
 */


export function isRetryableFailure(

    failure

) {


    return RETRYABLE_FAILURE_TYPES.has(

        failure?.failureType

    );

}
