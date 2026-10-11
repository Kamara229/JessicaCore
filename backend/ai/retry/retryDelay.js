/*
 * =========================================================
 * JESSICA AI RETRY
 * DELAY v1
 * =========================================================
 *
 * Отвечает только за:
 *
 * - sleep;
 * - вычисление delay.
 *
 *
 * НЕ:
 *
 * - классифицирует provider error;
 * - выполняет AI operation;
 * - управляет retry loop.
 *
 * =========================================================
 */


import {
    DEFAULT_RETRY_DELAY_MS,
    MAX_IMMEDIATE_RETRY_WAIT_MS,
    RETRY_SAFETY_MARGIN_MS
} from "./retryConstants.js";


import {
    getRetryAfterMilliseconds
} from "./retryAfter.js";


/*
 * =========================================================
 * SLEEP
 * =========================================================
 */


export function sleep(
    milliseconds
) {

    const delay =

        Number.isFinite(
            milliseconds
        )

            ? Math.max(
                0,
                milliseconds
            )

            : 0;


    return new Promise(
        resolve =>
            setTimeout(
                resolve,
                delay
            )
    );

}


/*
 * =========================================================
 * RETRY DELAY
 * =========================================================
 */


export function getAIRetryDelay(
    error,
    attempt
) {

    const retryAfter =

        getRetryAfterMilliseconds(
            error
        );


    /*
     * Provider value имеет приоритет
     * над нашим backoff.
     */


    if(
        retryAfter !== null
    ){

        return Math.min(

            retryAfter
            +
            RETRY_SAFETY_MARGIN_MS,

            MAX_IMMEDIATE_RETRY_WAIT_MS

        );

    }


    /*
     * Exponential backoff.
     *
     * attempt 1 → 1s
     * attempt 2 → 2s
     * attempt 3 → 4s
     */


    const normalizedAttempt =

        Number.isFinite(
            Number(attempt)
        )

            ? Math.max(
                1,
                Math.floor(
                    Number(attempt)
                )
            )

            : 1;


    const multiplier =

        normalizedAttempt - 1;


    return Math.min(

        DEFAULT_RETRY_DELAY_MS

        *

        Math.pow(
            2,
            multiplier
        ),

        MAX_IMMEDIATE_RETRY_WAIT_MS

    );

}
