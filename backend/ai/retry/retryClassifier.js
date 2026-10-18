/*
 * =========================================================
 * JESSICA AI RETRY
 * CLASSIFIER v2
 * =========================================================
 *
 * Определяет:
 *
 * provider / network error
 *      ↓
 * immediately retryable?
 *
 *
 * КРИТИЧЕСКО:
 *
 * Rate-limit errors обрабатываются
 * ДО общего x-should-retry veto.
 *
 *
 * Если provider явно пишет:
 *
 * "Please try again in 1.5525s"
 *
 * и ожидание входит в допустимый
 * immediate retry window,
 *
 * Jessica должна выполнить
 * локальный технический retry.
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
 * DAILY LIMIT
 * =========================================================
 *
 * Не пытаемся делать immediate retry
 * суточной квоты.
 * =========================================================
 */


function isDailyRateLimit(
    error
) {

    const message =

        getAIErrorMessage(
            error
        )
            .toLowerCase();


    return (

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

    );

}


/*
 * =========================================================
 * RATE LIMIT RETRY DECISION
 * =========================================================
 */


function isImmediatelyRetryableRateLimit(
    error
) {
    // An output budget larger than the provider's hard OTPM ceiling
    // cannot succeed on retry: the request shape must change.
    const details = getAIErrorMessage(error).toLowerCase();
    if (details.includes("request too large")
        || details.includes("expected output tokens exceed")
        || details.includes("reduce max_tokens")) return false;

    /*
     * Daily quota:
     *
     * immediate retry бессмысленен.
     */


    if(
        isDailyRateLimit(
            error
        )
    ){

        return false;

    }


    const retryAfter =

        getRetryAfterMilliseconds(
            error
        );


    /*
     * Provider дал конкретное время.
     *
     * Оно является главным критерием.
     */


    if(
        retryAfter !== null
    ){

        return (

            retryAfter
            <=
            MAX_IMMEDIATE_RETRY_WAIT_MS

        );

    }


    /*
     * 429 без конкретного времени.
     *
     * Разрешаем ограниченный backoff,
     * если provider явно не запретил retry.
     */


    return providerAllowsRetry(
        error
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
     * =====================================================
     * KNOWN MODEL PROTOCOL GLITCH
     * =====================================================
     */


    if(
        isTransientModelProtocolError(
            error
        )
    ){

        return true;

    }


    /*
     * =====================================================
     * RATE LIMIT
     * =====================================================
     *
     * Проверяем ДО providerAllowsRetry().
     *
     * Если provider прямо сказал:
     *
     * "try again in 1.55s"
     *
     * локальный retry допустим.
     * =====================================================
     */


    if(
        isRateLimitError(
            error
        )
    ){

        return isImmediatelyRetryableRateLimit(
            error
        );

    }


    /*
     * =====================================================
     * GENERIC PROVIDER VETO
     * =====================================================
     */


    if(
        !providerAllowsRetry(
            error
        )
    ){

        return false;

    }


    const status =

        getAIErrorStatus(
            error
        );


    /*
     * =====================================================
     * TIMEOUT
     * =====================================================
     */


    if(
        status === 408
    ){

        return true;

    }


    /*
     * =====================================================
     * 5XX
     * =====================================================
     */


    if(
        status >= 500
        &&
        status <= 599
    ){

        return true;

    }


    /*
     * =====================================================
     * NETWORK
     * =====================================================
     */


    return isRetryableNetworkError(
        error
    );

}
