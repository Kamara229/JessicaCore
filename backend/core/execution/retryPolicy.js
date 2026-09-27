/*
 * =========================================================
 * JESSICA RETRY POLICY v7
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
 * - делает Replan;
 * - вызывает Planner;
 * - выполняет Tools;
 * - оценивает качество результата.
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
 * RETRYABLE TYPES
 * =========================================================
 */


const RETRYABLE_TYPES = new Set([


    "temporary-error",


    "timeout-error",


    "network-error",


    "tool-error",


    "tool-exception",


    "runner-error",


    "runner-exception",


    "rate-limit-error",


    "service-unavailable"



]);









/*
 * =========================================================
 * RETRYABLE CATEGORIES
 * =========================================================
 */


const RETRYABLE_CATEGORIES = new Set([


    "temporary"

]);









/*
 * =========================================================
 * NORMALIZE
 * =========================================================
 */


function normalizeString(

    value

) {


    return String(

        value || ""

    )
    .toLowerCase()
    .trim();

}









function normalizeAttempt(

    value

) {


    const number =

        Number(value);



    return Number.isFinite(number)

        ?

        number

        :

        0;

}









/*
 * =========================================================
 * HELPERS
 * =========================================================
 */


function getFailureType(

    failure

) {


    return normalizeString(

        failure?.failureType

    );

}









function getFailureCategory(

    failure

) {


    return normalizeString(

        failure?.category

    );

}









/*
 * =========================================================
 * RETRY DECISION
 * =========================================================
 */


export function shouldRetryExecution(

    context,

    failure

) {


    const attempt =

        normalizeAttempt(

            context?.attempt

        );









    if (

        attempt >= MAX_EXECUTION_ATTEMPTS

    ) {


        return false;

    }









    if (

        !failure ||

        typeof failure !== "object"

    ) {


        return false;

    }









    /*
     * Пользователь должен ответить
     */


    if (

        failure.needsClarification === true

        ||

        failure.category === "clarification"

    ) {


        return false;

    }









    /*
     * Нет подтвержденного результата
     */


    if (

        failure.noVerifiedResult === true

        ||

        failure.category === "no_verified"

    ) {


        return false;

    }









    /*
     * Явный сигнал снизу
     */


    if (

        failure.shouldRetry === true

    ) {


        return true;

    }









    const type =

        getFailureType(

            failure

        );



    const category =

        getFailureCategory(

            failure

        );









    if (

        RETRYABLE_TYPES.has(

            type

        )

    ) {


        return true;

    }









    if (

        RETRYABLE_CATEGORIES.has(

            category

        )

    ) {


        return true;

    }









    return false;

}









/*
 * =========================================================
 * RETRY REASON
 * =========================================================
 */


export function getRetryReason(

    failure

) {


    const type =

        getFailureType(

            failure

        );



    switch(type){


        case "timeout-error":

            return "Временная ошибка ожидания";


        case "network-error":

            return "Ошибка сетевого соединения";


        case "tool-error":

        case "tool-exception":

            return "Ошибка выполнения инструмента";


        case "rate-limit-error":

            return "Превышен лимит запроса";


        case "service-unavailable":

            return "Сервис временно недоступен";


        default:

            return "Повтор выполнения маршрута";


    }

}









/*
 * =========================================================
 * STOP
 * =========================================================
 */


export function shouldStopExecution(

    context

) {


    return (

        normalizeAttempt(

            context?.attempt

        )

        >=

        MAX_EXECUTION_ATTEMPTS

    );

}









/*
 * =========================================================
 * REMAINING
 * =========================================================
 */


export function getRemainingAttempts(

    attempt

) {


    return Math.max(

        0,

        MAX_EXECUTION_ATTEMPTS -

        normalizeAttempt(

            attempt

        )

    );

}









/*
 * =========================================================
 * EXTERNAL CHECK
 * =========================================================
 */


export function isRetryableFailure(

    failure

) {


    return shouldRetryExecution(

        {

            attempt:0

        },

        failure

    );

}
