/*
 * =========================================================
 * JESSICA RETRY POLICY
 * =========================================================
 *
 * Единая политика повторных попыток выполнения подзадачи.
 *
 * Здесь хранятся только правила:
 *
 * - сколько попыток разрешено;
 * - можно ли делать retry;
 * - когда нужно остановиться.
 */


export const MAX_EXECUTION_ATTEMPTS = 3;


/*
 * =========================================================
 * SHOULD RETRY
 * =========================================================
 */


export function shouldRetryExecution(
    validation,
    attempt
) {

    if (
        attempt >=
        MAX_EXECUTION_ATTEMPTS
    ) {

        return false;
    }


    if (
        validation?.needsClarification === true
    ) {

        return false;
    }


    return (
        validation?.valid !== true &&
        validation?.shouldRetry === true
    );
}


/*
 * =========================================================
 * ATTEMPTS LEFT
 * =========================================================
 */


export function getRemainingAttempts(
    attempt
) {

    return Math.max(
        0,
        MAX_EXECUTION_ATTEMPTS - attempt
    );
}
