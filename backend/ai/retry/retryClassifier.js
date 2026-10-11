/*
 * =========================================================
 * JESSICA AI RETRY
 * CLASSIFIER v1
 * =========================================================
 *
 * Определяет:
 *
 * AI provider error
 *      ↓
 * retryable?
 *
 *
 * Поддерживает:
 *
 * - rate limit;
 * - timeout;
 * - provider 5xx;
 * - network errors;
 * - known temporary model protocol glitch.
 *
 *
 * НЕ:
 *
 * - вычисляет delay;
 * - делает sleep;
 * - выполняет retry loop.
 *
 * =========================================================
 */


import {
    MAX_IMMEDIATE_RETRY_WAIT_MS,
    RETRYABLE_NETWORK_CODES
} from "./retryConstants.js";


import {
    getAIErrorCode,
    getAIErrorHeader,
    getAIErrorMessage,
    getAIErrorStatus,
    getAINetworkErrorCode
} from "./retryError.js";


import {
    getRetryAfterMilliseconds
} from "./retryAfter.js";


/*
 * =========================================================
 * PROVIDER RETRY FLAG
 * =========================================================
 */


function providerAllowsRetry(
    error
) {

    const value =

        String(

            getAIErrorHeader(
                error,
                "x-should-retry"
            )

            ||

            ""

        )
            .trim()
            .toLowerCase();


    return value !== "false";

}


/*
 * =========================================================
 * TRANSIENT MODEL PROTOCOL ERROR
 * =========================================================
 *
 * Известный случай Groq + GPT OSS:
 *
 * HTTP 400
 *
 * "Tool choice is none,
 * but model called a tool"
 *
 *
 * Planner не передаёт executable tools.
 * Поэтому это не semantic error Plan.
 *
 * Ограниченный технический retry допустим.
 *
 * =========================================================
 */


export function isTransientModelProtocolError(
    error
) {

    if(
        getAIErrorStatus(
            error
        )
        !==
        400
    ){

        return false;

    }


    const message =

        getAIErrorMessage(
            error
        )
            .toLowerCase();


    return (

        message.includes(
            "tool choice is none"
        )

        &&

        message.includes(
            "model called a tool"
        )

    );

}


/*
 * =========================================================
 * RATE LIMIT
 * =========================================================
 */


function isRateLimitError(
    error
) {

    if(
        getAIErrorStatus(
            error
        )
        ===
        429
    ){

        return true;

    }


    if(
        getAIErrorCode(
            error
        )
        ===
        "rate_limit_exceeded"
    ){

        return true;

    }


    const message =

        getAIErrorMessage(
            error
        )
            .toLowerCase();


    return (

        message.includes(
            "rate limit"
        )

        ||

        message.includes(
            "tokens per minute"
        )

        ||

        message.includes(
            "(tpm)"
        )

        ||

        message.includes(
            "requests per minute"
        )

        ||

        message.includes(
            "(rpm)"
        )

    );

}


/*
 * =========================================================
 * LONG RATE LIMIT
 * =========================================================
 */


function isLongRateLimit(
    error
) {

    const message =

        getAIErrorMessage(
            error
        )
            .toLowerCase();


    /*
     * Daily quota.
     */


    if(
        message.includes(
            "tokens per day"
        )

        ||

        message.includes(
            "(tpd)"
        )

        ||

        message.includes(
            "requests per day"
        )

        ||

        message.includes(
            "(rpd)"
        )
    ){

        return true;

    }


    const retryAfter =

        getRetryAfterMilliseconds(
            error
        );


    return (

        retryAfter !== null

        &&

        retryAfter >
        MAX_IMMEDIATE_RETRY_WAIT_MS

    );

}


/*
 * =========================================================
 * NETWORK
 * =========================================================
 */


function isRetryableNetworkError(
    error
) {

    const code =

        getAINetworkErrorCode(
            error
        );


    return RETRYABLE_NETWORK_CODES.has(
        code
    );

}


/*
 * =========================================================
 * PUBLIC
 * =========================================================
 */


export function isRetryableAIError(
    error
) {

    /*
     * Known GPT-OSS protocol glitch.
     *
     * Проверяем отдельно,
     * потому что сам HTTP status = 400.
     */


    if(
        isTransientModelProtocolError(
            error
        )
    ){

        return true;

    }


    /*
     * Provider explicitly forbids retry.
     */


    if(
        !providerAllowsRetry(
            error
        )
    ){

        return false;

    }


    /*
     * Rate limit.
     */


    if(
        isRateLimitError(
            error
        )
    ){

        return !isLongRateLimit(
            error
        );

    }


    const status =

        getAIErrorStatus(
            error
        );


    /*
     * Request timeout.
     */


    if(
        status === 408
    ){

        return true;

    }


    /*
     * Temporary provider failure.
     */


    if(
        status >= 500
        &&
        status <= 599
    ){

        return true;

    }


    /*
     * Network.
     */


    return isRetryableNetworkError(
        error
    );

}
