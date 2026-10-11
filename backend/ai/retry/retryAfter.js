/*
 * =========================================================
 * JESSICA AI RETRY
 * RETRY-AFTER v1
 * =========================================================
 *
 * Извлекает provider retry delay из:
 *
 * - Retry-After header;
 * - Groq rate-limit reset headers;
 * - provider error message.
 *
 *
 * Поддерживает:
 *
 * 250ms
 * 11.19s
 * 1.5m
 * HTTP-date
 *
 *
 * НЕ:
 *
 * - определяет retryable;
 * - выполняет sleep;
 * - выполняет запрос.
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
     * Retry-After HTTP header
     * трактуется как seconds.
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
            Number.isFinite(amount)
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
            Number.isFinite(amount)
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
            Number.isFinite(amount)
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
     * Duration / seconds.
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
            String(value)
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
 * GROQ RESET HEADERS
 * =========================================================
 */


function getRateLimitResetFromHeaders(
    error
) {

    const headerNames = [

        "x-ratelimit-reset-tokens",

        "x-ratelimit-reset-requests"

    ];


    const delays = [];


    for(
        const headerName
        of headerNames
    ){

        const value =

            getAIErrorHeader(
                error,
                headerName
            );


        const duration =

            parseRetryDuration(
                value
            );


        if(
            duration !== null
        ){

            delays.push(
                duration
            );

        }

    }


    if(
        delays.length === 0
    ){

        return null;

    }


    /*
     * Если действуют одновременно
     * несколько ограничений,
     * ждём наиболее консервативное.
     */


    return Math.max(
        ...delays
    );

}


/*
 * =========================================================
 * MESSAGE
 * =========================================================
 *
 * Пример Groq:
 *
 * "Please try again in 11.19s"
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
        !Number.isFinite(amount)
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
            .includes(unit)
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
 * PUBLIC
 * =========================================================
 */


export function getRetryAfterMilliseconds(
    error
) {

    const values = [

        getRetryAfterFromHeader(
            error
        ),

        getRateLimitResetFromHeaders(
            error
        ),

        getRetryAfterFromMessage(
            error
        )

    ]
        .filter(
            value =>
                Number.isFinite(value)
                &&
                value >= 0
        );


    if(
        values.length === 0
    ){

        return null;

    }


    return Math.max(
        ...values
    );

}
