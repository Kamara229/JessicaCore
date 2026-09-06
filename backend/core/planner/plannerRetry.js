/*
 * =========================================================
 * JESSICA PLANNER RETRY
 * =========================================================
 *
 * Техническая логика повторных попыток Planner.
 *
 * Здесь нет логики планирования задач.
 */


export const MAX_PLANNER_ATTEMPTS = 3;


/*
 * =========================================================
 * SLEEP
 * =========================================================
 */


export function sleep(ms) {

    return new Promise(
        resolve =>
            setTimeout(resolve, ms)
    );

}


/*
 * =========================================================
 * RETRYABLE ERROR
 * =========================================================
 */


export function isRetryablePlannerError(error) {

    if (!error) {
        return false;
    }


    const status =
        Number(
            error.status ||
            error.statusCode ||
            error.response?.status ||
            0
        );


    /*
     * Rate limit.
     */
    if (status === 429) {
        return true;
    }


    /*
     * Временные ошибки сервера AI.
     */
    if (
        status >= 500 &&
        status <= 599
    ) {
        return true;
    }


    const code =
        String(
            error.code || ""
        ).toUpperCase();


    const retryableCodes =
        new Set([
            "ETIMEDOUT",
            "ECONNRESET",
            "ECONNREFUSED",
            "EAI_AGAIN",
            "ENETUNREACH"
        ]);


    if (
        retryableCodes.has(code)
    ) {
        return true;
    }


    const message =
        String(
            error.message || ""
        ).toLowerCase();


    return (
        message.includes("timeout") ||
        message.includes("timed out") ||
        message.includes("network") ||
        message.includes("rate limit") ||
        message.includes("temporarily unavailable")
    );

}


/*
 * =========================================================
 * RETRY DELAY
 * =========================================================
 */


export function getPlannerRetryDelay(
    attempt
) {

    /*
     * 1-я повторная попытка: 700 мс
     * 2-я: 1400 мс
     * 3-я: 2800 мс
     */

    const safeAttempt =
        Math.max(
            1,
            Number(attempt) || 1
        );


    return Math.min(
        700 *
            Math.pow(
                2,
                safeAttempt - 1
            ),
        5000
    );

}
