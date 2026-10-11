/*
 * =========================================================
 * JESSICA RETRY POLICY v9
 * =========================================================
 *
 * Политика повторных попыток Execution Route.
 *
 *
 * Отвечает только:
 *
 * - можно ли повторить текущий маршрут;
 * - сколько попыток осталось.
 *
 *
 * ВАЖНО:
 *
 * AI transport retry
 *      !=
 * Execution retry.
 *
 *
 * Presentation failure
 * не должен повторять Tools.
 *
 * =========================================================
 */


export const MAX_EXECUTION_ATTEMPTS =
    3;


/*
 * =========================================================
 * RETRYABLE TYPES
 * =========================================================
 */


const RETRYABLE_TYPES =
    new Set([

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


const RETRYABLE_CATEGORIES =
    new Set([

        "temporary",

        "tool"

    ]);


/*
 * =========================================================
 * NON RETRYABLE
 * =========================================================
 */


const NON_RETRYABLE_CATEGORIES =
    new Set([

        "clarification",

        "no_verified",

        "planner",

        "data",

        "validation",

        /*
         * Composer / presentation failure
         * не должен повторять Execution.
         */

        "presentation"

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

        ? number

        : 0;

}


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
 * SHOULD RETRY
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


    if(
        attempt >=
        MAX_EXECUTION_ATTEMPTS
    ){

        return false;

    }


    if(
        !failure
        ||
        typeof failure !== "object"
    ){

        return false;

    }


    if(
        failure.needsClarification
        ===
        true
    ){

        return false;

    }


    if(
        failure.noVerifiedResult
        ===
        true
    ){

        return false;

    }


    const type =

        getFailureType(
            failure
        );


    const category =

        getFailureCategory(
            failure
        );


    /*
     * =====================================================
     * EXPLICIT NON-RETRY
     * =====================================================
     */


    if(
        NON_RETRYABLE_CATEGORIES.has(
            category
        )
    ){

        return false;

    }


    /*
     * =====================================================
     * TYPE
     * =====================================================
     */


    if(
        RETRYABLE_TYPES.has(
            type
        )
    ){

        return true;

    }


    /*
     * =====================================================
     * CATEGORY
     * =====================================================
     */


    if(
        RETRYABLE_CATEGORIES.has(
            category
        )
    ){

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

        MAX_EXECUTION_ATTEMPTS

        -

        normalizeAttempt(
            attempt
        )

    );

}


/*
 * =========================================================
 * HELPER
 * =========================================================
 */


export function isRetryableFailure(
    failure
) {

    return shouldRetryExecution(

        {
            attempt:
                0
        },

        failure

    );

}
