/*
 * =========================================================
 * JESSICA PLANNER RETRY POLICY v3
 * =========================================================
 *
 * Техническая политика повторных попыток Planner.
 *
 *
 * Отвечает:
 *
 * - количество попыток;
 * - задержки;
 * - определение временных ошибок.
 *
 *
 * НЕ:
 *
 * - создаёт план;
 * - вызывает AI;
 * - исправляет JSON;
 * - анализирует Experience.
 *
 * =========================================================
 */





/*
 * =========================================================
 * CONFIG
 * =========================================================
 */


export const MAX_PLANNER_ATTEMPTS =
    3;




const INITIAL_DELAY =
    700;



const MAX_DELAY =
    5000;









/*
 * =========================================================
 * SLEEP
 * =========================================================
 */


export function sleep(
    milliseconds
) {


    return new Promise(

        resolve =>
            setTimeout(
                resolve,
                milliseconds
            )

    );

}









/*
 * =========================================================
 * ERROR HELPERS
 * =========================================================
 */


function getStatus(
    error
) {


    return Number(

        error?.status ||

        error?.statusCode ||

        error?.response?.status ||

        0

    );


}






function getCode(
    error
) {


    return String(

        error?.code ||

        ""

    )
    .toUpperCase();


}






function getMessage(
    error
) {


    return String(

        error?.message ||

        error ||

        ""

    )
    .toLowerCase();


}









/*
 * =========================================================
 * NETWORK ERRORS
 * =========================================================
 */


const NETWORK_CODES =
    new Set([

        "ETIMEDOUT",

        "ECONNRESET",

        "ECONNREFUSED",

        "EAI_AGAIN",

        "ENETUNREACH"

    ]);









/*
 * =========================================================
 * RETRYABLE ERROR
 * =========================================================
 */


export function isRetryablePlannerError(
    error
) {


    if (
        !error
    ) {

        return false;

    }



    const status =
        getStatus(
            error
        );



    /*
     * Rate limit
     */


    if (
        status === 429
    ) {

        return true;

    }





    /*
     * Временные ошибки AI сервера
     */


    if (
        status >= 500 &&
        status <= 599
    ) {

        return true;

    }





    /*
     * Network
     */


    const code =
        getCode(
            error
        );


    if (
        NETWORK_CODES.has(
            code
        )
    ) {

        return true;

    }





    const message =
        getMessage(
            error
        );






    /*
     * Timeout
     */


    if (

        message.includes(
            "timeout"
        )

        ||

        message.includes(
            "timed out"
        )

    ) {

        return true;

    }






    /*
     * AI временно недоступен
     */


    if (

        message.includes(
            "temporarily unavailable"
        )

        ||

        message.includes(
            "service unavailable"
        )

        ||

        message.includes(
            "network"
        )

    ) {

        return true;

    }








    /*
     * Ошибки формата ответа AI.
     *
     * Иногда модель возвращает
     * неправильный JSON.
     */


    if (

        message.includes(
            "json"
        )

        ||

        message.includes(
            "parse"
        )

        ||

        message.includes(
            "normalize"
        )

        ||

        message.includes(
            "planner validation"
        )

    ) {

        return true;

    }





    return false;


}









/*
 * =========================================================
 * RETRY DELAY
 * =========================================================
 */


export function getPlannerRetryDelay(
    attempt
) {


    const safeAttempt =
        Math.max(

            1,

            Number(
                attempt
            )
            ||
            1

        );



    return Math.min(

        INITIAL_DELAY *
        Math.pow(

            2,

            safeAttempt - 1

        ),

        MAX_DELAY

    );


}









/*
 * =========================================================
 * RETRY INFORMATION
 * =========================================================
 *
 * Используется для логов
 * и Trace.
 *
 * =========================================================
 */


export function getRetryInfo(
    attempt,
    error
) {


    return {


        attempt:


            Number(
                attempt || 0
            ),



        maxAttempts:


            MAX_PLANNER_ATTEMPTS,



        retryable:


            isRetryablePlannerError(
                error
            ),



        delay:


            getPlannerRetryDelay(
                attempt
            ),



        reason:


            getMessage(
                error
            )


    };


}
