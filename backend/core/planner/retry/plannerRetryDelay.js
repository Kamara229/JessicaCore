/*
 * =========================================================
 * JESSICA PLANNER RETRY DELAY
 * =========================================================
 *
 * Рассчитывает фактическую задержку retry.
 *
 * Приоритет:
 *
 * provider Retry-After
 *      ↓
 * exponential fallback
 *
 * =========================================================
 */


import {

    INITIAL_RETRY_DELAY,

    MAX_RETRY_DELAY,

    RETRY_AFTER_BUFFER,

    MAX_RETRY_AFTER

} from "./plannerRetryConfig.js";


import {
    getPlannerRetryAfter
} from "./plannerRetryAfter.js";


function getFallbackRetryDelay(
    attempt
) {


    const safeAttempt =
        Math.max(

            1,

            Number(attempt) || 1

        );


    return Math.min(

        INITIAL_RETRY_DELAY *
        Math.pow(
            2,
            safeAttempt - 1
        ),

        MAX_RETRY_DELAY

    );

}


export function getPlannerRetryDelay(
    attempt,
    error = null
) {


    const retryAfter =
        getPlannerRetryAfter(
            error
        );


    if (
        retryAfter !== null
    ) {


        return Math.min(

            retryAfter +
            RETRY_AFTER_BUFFER,

            MAX_RETRY_AFTER

        );

    }


    return getFallbackRetryDelay(
        attempt
    );

}
