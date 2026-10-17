/*
 * =========================================================
 * JESSICA PLANNER RETRY ERRORS v2
 * =========================================================
 *
 * Классификация ошибок Planner:
 *
 * - technical;
 * - semantic;
 * - retryable.
 *
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
 * ERROR DATA
 * =========================================================
 */


export function getPlannerErrorStatus(
    error
) {


    return Number(

        error?.status ||

        error?.statusCode ||

        error?.response?.status ||

        0

    );

}


export function getPlannerErrorCode(
    error
) {


    return String(

        error?.code ||

        ""

    )
    .toUpperCase();

}


export function getPlannerErrorMessage(
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
 * TECHNICAL ERROR
 * =========================================================
 */


export function isTechnicalPlannerError(
    error
) {


    if (
        !error
    ) {

        return false;

    }


    const status =
        getPlannerErrorStatus(
            error
        );


    if (
        status === 429
    ) {

        return true;

    }


    if (
        status >= 500 &&
        status <= 599
    ) {

        return true;

    }


    const code =
        getPlannerErrorCode(
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
        getPlannerErrorMessage(
            error
        );


    return (

        message.includes(
            "timeout"
        )

        ||

        message.includes(
            "timed out"
        )

        ||

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

        ||

        message.includes(
            "rate limit"
        )

    );

}


/*
 * =========================================================
 * SEMANTIC ERROR
 * =========================================================
 */


export function isSemanticPlannerError(
    error
) {


    if (
        !error
    ) {

        return false;

    }


    if (
        isTechnicalPlannerError(
            error
        )
    ) {

        return false;

    }


    const message =
        getPlannerErrorMessage(
            error
        );


    return (

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

        ||

        message.includes(
            "planner_parse_failed"
        )

        ||

        message.includes(
            "planner_normalize_failed"
        )

    );

}


/*
 * =========================================================
 * RETRYABLE
 * =========================================================
 */


export function isRetryablePlannerError(
    error
) {


    /*
     * Technical provider/network retries
     * уже принадлежат AI transport layer:
     *
     * plannerClient
     *   ↓
     * executeAIWithRetry
     *
     * Если technical error вышла наружу,
     * локальный AI retry уже либо исчерпан,
     * либо provider прямо запретил immediate retry.
     *
     * Planner attempt loop НЕ должен запускать
     * второй независимый technical retry loop.
     *
     * Здесь повторяем только semantic failures
     * самого Planner pipeline:
     *
     * parse / normalize / semantic validation.
     */


    return isSemanticPlannerError(
        error
    );

}
