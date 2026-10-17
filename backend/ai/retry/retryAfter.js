/*
 * =========================================================
 * JESSICA AI RETRY
 * RETRY-AFTER v2
 * =========================================================
 *
 * Извлекает provider retry delay.
 *
 *
 * PRIORITY:
 *
 * 1. Retry-After header
 * 2. explicit provider message:
 *      "Please try again in 1.5525s"
 * 3. rate-limit reset header,
 *    соответствующий реально сработавшему limit
 * 4. conservative reset fallback
 *
 *
 * ВАЖНО:
 *
 * x-ratelimit-reset-tokens
 * и
 * x-ratelimit-reset-requests
 *
 * не являются автоматически
 * одновременно активными ограничениями.
 *
 * Поэтому нельзя всегда брать Math.max()
 * между ними и explicit provider message.
 *
 * =========================================================
 */


import {
    getAIErrorHeader,
    getAIErrorMessage,
    safeRetryString
} from "./retryError.js";


/*
 * =========================================================
 * DURATION
 * =========================================================
 */


export function parseRetryDuration(
    value
) {

    const text =

        safeRetryString(
            value
        )
            .toLowerCase();


    if(
        !text
    ){

        return null;

    }


    /*
     * Plain number.
     *
     * HTTP Retry-After without unit
     * means seconds.
     */


    const plainNumber =

        Number(
            text
        );


    if(
        Number.isFinite(
            plainNumber
        )
        &&
        plainNumber >= 0
    ){

        return Math.ceil(
            plainNumber * 1000
        );

    }


    /*
     * Milliseconds.
     */


    const millisecondsMatch =

        text.match(
            /^([\d.]+)\s*ms$/
        );


    if(
        millisecondsMatch
    ){

        const amount =

            Number(
                millisecondsMatch[1]
            );


        if(
            Number.isFinite(
                amount
            )
            &&
            amount >= 0
        ){

            return Math.ceil(
                amount
            );

        }

    }


    /*
     * Seconds.
     */


    const secondsMatch =

        text.match(
            /^([\d.]+)\s*(?:s|sec|secs|second|seconds)$/
        );


    if(
        secondsMatch
    ){

        const amount =

            Number(
                secondsMatch[1]
            );


        if(
            Number.isFinite(
                amount
            )
            &&
            amount >= 0
        ){

            return Math.ceil(
                amount * 1000
            );

        }

    }


    /*
     * Minutes.
     */


    const minutesMatch =

        text.match(
            /^([\d.]+)\s*(?:m|min|mins|minute|minutes)$/
        );


    if(
        minutesMatch
    ){

        const amount =

            Number(
                minutesMatch[1]
            );


        if(
            Number.isFinite(
                amount
            )
            &&
            amount >= 0
        ){

            return Math.ceil(
                amount * 60_000
            );

        }

    }


    return null;

}


/*
 * =========================================================
 * RETRY-AFTER HEADER
 * =========================================================
 */


function getRetryAfterFromHeader(
    error
) {

    const value =

        getAIErrorHeader(
            error,
            "retry-after"
        );


    if(
        value === null
        ||
        value === undefined
        ||
        value === ""
    ){

        return null;

    }


    /*
     * Seconds / duration.
     */


    const duration =

        parseRetryDuration(
            value
        );


    if(
        duration !== null
    ){

        return duration;

    }


    /*
     * HTTP-date.
     */


    const timestamp =

        Date.parse(
            String(
                value
            )
        );


    if(
        Number.isFinite(
            timestamp
        )
    ){

        return Math.max(

            0,

            timestamp - Date.now()

        );

    }


    return null;

}


/*
 * =========================================================
 * PROVIDER MESSAGE
 * =========================================================
 *
 * Example:
 *
 * Please try again in 1.5525s
 *
 * =========================================================
 */


function getRetryAfterFromMessage(
    error
) {

    const message =

        getAIErrorMessage(
            error
        );


    if(
        !message
    ){

        return null;

    }


    const match =

        message.match(

            /try\s+again\s+in\s+([\d.]+)\s*(ms|s|sec|secs|second|seconds|m|min|mins|minute|minutes)\b/i

        );


    if(
        !match
    ){

        return null;

    }


    const amount =

        Number(
            match[1]
        );


    if(
        !Number.isFinite(
            amount
        )
        ||
        amount < 0
    ){

        return null;

    }


    const unit =

        String(
            match[2]
        )
            .trim()
            .toLowerCase();


    if(
        unit === "ms"
    ){

        return Math.ceil(
            amount
        );

    }


    if(
        [
            "m",
            "min",
            "mins",
            "minute",
            "minutes"
        ]
            .includes(
                unit
            )
    ){

        return Math.ceil(
            amount * 60_000
        );

    }


    return Math.ceil(
        amount * 1000
    );

}


/*
 * =========================================================
 * RESET HEADER
 * =========================================================
 */


function getResetHeaderDelay(
    error,
    headerName
) {

    const value =

        getAIErrorHeader(
            error,
            headerName
        );


    return parseRetryDuration(
        value
    );

}


/*
 * =========================================================
 * RATE LIMIT DIMENSION
 * =========================================================
 */


function detectRateLimitDimension(
    error
) {

    const message =

        getAIErrorMessage(
            error
        )
            .toLowerCase();


    /*
     * Token limit.
     */


    if(
        message.includes(
            "tokens per minute"
        )
        ||
        message.includes(
            "(tpm)"
        )
        ||
        message.includes(
            "tokens per day"
        )
        ||
        message.includes(
            "(tpd)"
        )
    ){

        return "tokens";

    }


    /*
     * Request limit.
     */


    if(
        message.includes(
            "requests per minute"
        )
        ||
        message.includes(
            "(rpm)"
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

        return "requests";

    }


    return null;

}


/*
 * =========================================================
 * DETAILS
 * =========================================================
 */


export function getRetryAfterDetails(
    error
) {

    /*
     * =====================================================
     * 1. STANDARD RETRY-AFTER
     * =====================================================
     */


    const retryAfterHeader =

        getRetryAfterFromHeader(
            error
        );


    if(
        retryAfterHeader !== null
    ){

        return {

            milliseconds:
                retryAfterHeader,

            source:
                "retry-after"

        };

    }


    /*
     * =====================================================
     * 2. EXPLICIT PROVIDER MESSAGE
     * =====================================================
     *
     * Если provider сам написал:
     *
     * "Please try again in 1.5525s"
     *
     * это наиболее конкретная инструкция
     * для текущего отказа.
     * =====================================================
     */


    const messageDelay =

        getRetryAfterFromMessage(
            error
        );


    if(
        messageDelay !== null
    ){

        return {

            milliseconds:
                messageDelay,

            source:
                "provider-message"

        };

    }


    /*
     * =====================================================
     * 3. LIMIT-SPECIFIC RESET HEADER
     * =====================================================
     */


    const tokenReset =

        getResetHeaderDelay(

            error,

            "x-ratelimit-reset-tokens"

        );


    const requestReset =

        getResetHeaderDelay(

            error,

            "x-ratelimit-reset-requests"

        );


    const dimension =

        detectRateLimitDimension(
            error
        );


    if(
        dimension === "tokens"
        &&
        tokenReset !== null
    ){

        return {

            milliseconds:
                tokenReset,

            source:
                "x-ratelimit-reset-tokens"

        };

    }


    if(
        dimension === "requests"
        &&
        requestReset !== null
    ){

        return {

            milliseconds:
                requestReset,

            source:
                "x-ratelimit-reset-requests"

        };

    }


    /*
     * =====================================================
     * 4. FALLBACK
     * =====================================================
     *
     * Только когда неизвестно,
     * какое именно ограничение сработало.
     *
     * Здесь консервативный max допустим.
     * =====================================================
     */


    const available = [

        tokenReset !== null

            ? {

                milliseconds:
                    tokenReset,

                source:
                    "x-ratelimit-reset-tokens"

            }

            : null,


        requestReset !== null

            ? {

                milliseconds:
                    requestReset,

                source:
                    "x-ratelimit-reset-requests"

            }

            : null

    ]
        .filter(Boolean);


    if(
        available.length === 0
    ){

        return null;

    }


    return available.reduce(

        (
            longest,
            current
        ) => {

            if(
                !longest
                ||
                current.milliseconds
                >
                longest.milliseconds
            ){

                return current;

            }


            return longest;

        },

        null

    );

}


/*
 * =========================================================
 * PUBLIC DELAY
 * =========================================================
 */


export function getRetryAfterMilliseconds(
    error
) {

    const details =

        getRetryAfterDetails(
            error
        );


    return details
        ?.milliseconds
        ??
        null;

}
