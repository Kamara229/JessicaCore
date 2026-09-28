/*
 * =========================================================
 * JESSICA PLANNER RETRY POLICY v4
 * =========================================================
 *
 * Центральный публичный интерфейс retry-системы Planner.
 *
 *
 * Внутренние модули:
 *
 * retry/plannerRetryConfig.js
 * retry/plannerRetryErrors.js
 * retry/plannerRetryAfter.js
 * retry/plannerRetryDelay.js
 *
 *
 * Остальной Jessica Core должен работать
 * только через этот файл.
 *
 * =========================================================
 */


import {
    MAX_PLANNER_ATTEMPTS
} from "./retry/plannerRetryConfig.js";


import {

    getPlannerErrorMessage,

    isTechnicalPlannerError,

    isSemanticPlannerError,

    isRetryablePlannerError

} from "./retry/plannerRetryErrors.js";


import {
    getPlannerRetryAfter
} from "./retry/plannerRetryAfter.js";


import {
    getPlannerRetryDelay
} from "./retry/plannerRetryDelay.js";


/*
 * =========================================================
 * PUBLIC CONFIG
 * =========================================================
 */


export {
    MAX_PLANNER_ATTEMPTS
};


/*
 * =========================================================
 * PUBLIC ERROR CLASSIFICATION
 * =========================================================
 */


export {

    isTechnicalPlannerError,

    isSemanticPlannerError,

    isRetryablePlannerError

};


/*
 * =========================================================
 * PUBLIC RETRY-AFTER
 * =========================================================
 */


export {
    getPlannerRetryAfter
};


/*
 * =========================================================
 * PUBLIC DELAY
 * =========================================================
 */


export {
    getPlannerRetryDelay
};


/*
 * =========================================================
 * SLEEP
 * =========================================================
 */


export function sleep(
    milliseconds
) {


    const delay =
        Math.max(

            0,

            Number(milliseconds) || 0

        );


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
 * RETRY INFORMATION
 * =========================================================
 */


export function getRetryInfo(
    attempt,
    error
) {


    const technical =
        isTechnicalPlannerError(
            error
        );


    const semantic =
        isSemanticPlannerError(
            error
        );


    const retryAfter =
        getPlannerRetryAfter(
            error
        );


    return {

        attempt:
            Number(
                attempt || 0
            ),

        maxAttempts:
            MAX_PLANNER_ATTEMPTS,

        retryable:
            technical || semantic,

        type:
            technical
                ? "technical"
                : semantic
                    ? "semantic"
                    : "non-retryable",

        retryAfter,

        delay:
            getPlannerRetryDelay(
                attempt,
                error
            ),

        reason:
            getPlannerErrorMessage(
                error
            )

    };

}
