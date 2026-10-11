/*
 * =========================================================
 * JESSICA AI RETRY v2
 * =========================================================
 *
 * Общая техническая политика повторов AI-запросов.
 *
 *
 * Повторяем только временные provider failures:
 *
 * - TPM / RPM rate limit;
 * - timeout;
 * - network;
 * - 5xx.
 *
 *
 * НЕ повторяем:
 *
 * - semantic failures;
 * - invalid request;
 * - долгосрочные quota limits;
 * - provider x-should-retry=false.
 *
 * =========================================================
 */


/*
 * =========================================================
 * CONFIG
 * =========================================================
 */


export const MAX_AI_ATTEMPTS =
    3;


const DEFAULT_RETRY_DELAY_MS =
    1000;


const MAX_IMMEDIATE_RETRY_WAIT_MS =
    30_000;


const RETRY_SAFETY_MARGIN_MS =
    350;


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
 * SAFE STRING
 * =========================================================
 */


function safeString(
    value
) {

    return typeof value === "string"

        ? value.trim()

        : "";

}


/*
 * =========================================================
 * ERROR MESSAGE
 * =========================================================
 */


function getErrorMessage(
    error
) {

    return safeString(

        error?.message

        ||

        error?.error?.message

        ||

        error?.response?.data?.error?.message

        ||

        error?.response?.data?.message

    );

}


/*
 * =========================================================
 * HEADER
 * =========================================================
 */


function getHeader(
    error,
    name
) {

    const headers =
        error?.headers;


    if(
        !headers
    ){

        return null;

    }


    try {


        if(
            typeof headers.get === "function"
        ){

            return headers.get(
                name
            );

        }


        return (

            headers[name]

            ??

            headers[
                String(name)
                    .toLowerCase()
            ]

            ??

            null

        );


    }catch(error){


        return null;

    }

}


/*
 * =========================================================
 * STATUS
 * =========================================================
 */


function getErrorStatus(
    error
) {

    const candidates = [

        error?.status,

        error?.statusCode,

        error?.response?.status

    ];


    for(
        const candidate
        of candidates
    ){

        const status =
            Number(candidate);


        if(
            Number.isFinite(
                status
            )
            &&
            status > 0
        ){

            return status;

        }

    }


    return 0;

}


/*
 * =========================================================
 * ERROR CODE
 * =========================================================
 */


function getErrorCode(
    error
) {

    return safeString(

        error?.code

        ||

        error?.error?.code

    )
        .toLowerCase();

}


/*
 * =========================================================
 * PROVIDER RETRY POLICY
 * =========================================================
 */


function providerAllowsRetry(
    error
) {

    const value =

        safeString(

            getHeader(
                error,
                "x-should-retry"
            )

        )
            .toLowerCase();


    if(
        value === "false"
    ){

        return false;

    }


    return true;

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

        getHeader(
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
     * Seconds
     */


    const seconds =
        Number(value);


    if(
        Number.isFinite(
            seconds
        )
        &&
        seconds >= 0
    ){

        return Math.ceil(
            seconds * 1000
        );

    }


    /*
     * HTTP date
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
 * RETRY-AFTER MESSAGE
 * =========================================================
 *
 * Groq иногда сообщает delay только текстом:
 *
 * "Please try again in 11.19s"
 *
 * Поэтому одного HTTP header недостаточно.
 *
 * =========================================================
 */


function getRetryAfterFromMessage(
    error
) {

    const message =

        getErrorMessage(
            error
        );


    if(
        !message
    ){

        return null;

    }


    /*
     * milliseconds
     */


    const millisecondsMatch =

        message.match(
            /try\s+again\s+in\s+([\d.]+)\s*ms\b/i
        );


    if(
        millisecondsMatch
    ){

        const value =

            Number(
                millisecondsMatch[1]
            );


        if(
            Number.isFinite(value)
            &&
            value >= 0
        ){

            return Math.ceil(
                value
            );

        }

    }


    /*
     * seconds
     */


    const secondsMatch =

        message.match(
            /try\s+again\s+in\s+([\d.]+)\s*(?:s|sec|secs|second|seconds)\b/i
        );


    if(
        secondsMatch
    ){

        const value =

            Number(
                secondsMatch[1]
            );


        if(
            Number.isFinite(value)
            &&
            value >= 0
        ){

            return Math.ceil(
                value * 1000
            );

        }

    }


    /*
     * minutes
     */


    const minutesMatch =

        message.match(
            /try\s+again\s+in\s+([\d.]+)\s*(?:m|min|mins|minute|minutes)\b/i
        );


    if(
        minutesMatch
    ){

        const value =

            Number(
                minutesMatch[1]
            );


        if(
            Number.isFinite(value)
            &&
            value >= 0
        ){

            return Math.ceil(
                value * 60_000
            );

        }

    }


    return null;

}


/*
 * =========================================================
 * RETRY-AFTER
 * =========================================================
 */


function getRetryAfterMilliseconds(
    error
) {

    const headerValue =

        getRetryAfterFromHeader(
            error
        );


    if(
        headerValue !== null
    ){

        return headerValue;

    }


    return getRetryAfterFromMessage(
        error
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
        getErrorStatus(
            error
        )
        ===
        429
    ){

        return true;

    }


    const code =

        getErrorCode(
            error
        );


    if(
        code === "rate_limit_exceeded"
    ){

        return true;

    }


    const message =

        getErrorMessage(
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
 * LONG-LIVED RATE LIMIT
 * =========================================================
 */


function isLongRateLimit(
    error
) {

    const message =

        getErrorMessage(
            error
        )
            .toLowerCase();


    /*
     * Daily / long quota.
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


    if(
        retryAfter !== null
        &&
        retryAfter >
        MAX_IMMEDIATE_RETRY_WAIT_MS
    ){

        return true;

    }


    return false;

}


/*
 * =========================================================
 * RETRYABLE ERROR
 * =========================================================
 */


export function isRetryableAIError(
    error
) {

    /*
     * Provider explicit prohibition.
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

        if(
            isLongRateLimit(
                error
            )
        ){

            return false;

        }


        return true;

    }


    const status =

        getErrorStatus(
            error
        );


    /*
     * Request timeout.
     */


    if(
        status === 408
        ||
        status === 408
    ){

        return true;

    }


    /*
     * Temporary provider errors.
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


    const code =

        safeString(
            error?.code
        )
            .toUpperCase();


    const retryableCodes =

        new Set([

            "ECONNRESET",

            "ECONNREFUSED",

            "ETIMEDOUT",

            "EAI_AGAIN",

            "ENETUNREACH",

            "UND_ERR_CONNECT_TIMEOUT",

            "UND_ERR_HEADERS_TIMEOUT",

            "UND_ERR_SOCKET"

        ]);


    return retryableCodes.has(
        code
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
     * attempt 1 → 1 sec
     * attempt 2 → 2 sec
     * attempt 3 → 4 sec
     */


    const multiplier =

        Math.max(
            0,
            attempt - 1
        );


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


/*
 * =========================================================
 * EXECUTE WITH RETRY
 * =========================================================
 */


export async function executeAIWithRetry(

    operation,

    options = {}

) {

    if(
        typeof operation !== "function"
    ){

        throw new Error(
            "AI retry operation не является функцией"
        );

    }


    const label =

        safeString(
            options.label
        )

        ||

        "AI request";


    const maxAttempts =

        Number.isInteger(
            options.maxAttempts
        )
        &&
        options.maxAttempts > 0

            ? options.maxAttempts

            : MAX_AI_ATTEMPTS;


    let lastError =
        null;


    for(
        let attempt = 1;

        attempt <= maxAttempts;

        attempt++
    ){

        try {


            return await operation();


        }catch(error){


            lastError =
                error;


            const retryable =

                isRetryableAIError(
                    error
                );


            const delay =

                retryable
                &&
                attempt < maxAttempts

                    ? getAIRetryDelay(
                        error,
                        attempt
                    )

                    : 0;


            console.warn(

                `${label} failed [${attempt}/${maxAttempts}]:`,

                {

                    status:

                        getErrorStatus(
                            error
                        ),

                    code:

                        getErrorCode(
                            error
                        )

                        ||

                        null,

                    retryable,

                    retryDelay:
                        delay,

                    message:

                        getErrorMessage(
                            error
                        )

                        ||

                        String(error)

                }

            );


            /*
             * =================================================
             * DO NOT RETRY
             * =================================================
             */


            if(
                !retryable
            ){

                console.warn(

                    `${label}: provider error is not immediately retryable`

                );


                throw error;

            }


            /*
             * =================================================
             * ATTEMPTS EXHAUSTED
             * =================================================
             */


            if(
                attempt >= maxAttempts
            ){

                throw error;

            }


            /*
             * =================================================
             * WAIT
             * =================================================
             */


            console.log(

                `${label} technical retry in ${delay}ms`

            );


            await sleep(
                delay
            );

        }

    }


    throw (

        lastError

        ||

        new Error(
            `${label} failed`
        )

    );

}
