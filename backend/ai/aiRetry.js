/*
 * =========================================================
 * JESSICA AI RETRY v3
 * =========================================================
 *
 * Общая техническая политика повторов AI-запросов.
 *
 *
 * Повторяем только временные provider failures:
 *
 * - TPM / RPM rate limit;
 * - timeout;
 * - network failures;
 * - 5xx;
 * - известный временный Groq/model protocol glitch.
 *
 *
 * НЕ повторяем:
 *
 * - semantic failures;
 * - обычные 400 invalid request;
 * - долгосрочные quota limits;
 * - provider x-should-retry=false,
 *   кроме явно известного model protocol glitch.
 *
 *
 * Semantic retry здесь НЕ выполняется.
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


/*
 * Небольшой запас после provider Retry-After.
 *
 * Если Groq говорит 11.19s,
 * запрос уйдёт примерно через 11.54s,
 * а не ровно на границе rate-limit окна.
 */


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
            Number.isFinite(status)
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


        const normalizedName =

            String(name)
                .toLowerCase();


        return (

            headers[name]

            ??

            headers[normalizedName]

            ??

            null

        );


    }catch(error){


        return null;

    }

}


/*
 * =========================================================
 * KNOWN MODEL PROTOCOL ERROR
 * =========================================================
 *
 * Groq + GPT OSS иногда возвращает:
 *
 * 400 Tool choice is none,
 * but model called a tool
 *
 *
 * При этом Jessica Planner:
 *
 * - не передаёт tools;
 * - не разрешает tool execution;
 * - ожидает обычный JSON.
 *
 *
 * Поэтому это не ошибка Execution Plan
 * и не повод запускать Replanner.
 *
 * Это ограниченно повторяемый transport /
 * model-generation glitch.
 *
 * =========================================================
 */


function isTransientModelProtocolError(
    error
) {

    const status =

        getErrorStatus(
            error
        );


    if(
        status !== 400
    ){

        return false;

    }


    const message =

        getErrorMessage(
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
 * RETRY AFTER FROM HEADER
 * =========================================================
 */


function parseRetryDuration(
    value
) {

    const text =

        safeString(
            value
        )
            .toLowerCase();


    if(
        !text
    ){

        return null;

    }


    /*
     * Plain seconds:
     *
     * Retry-After: 11.19
     */


    const plainNumber =
        Number(text);


    if(
        Number.isFinite(plainNumber)
        &&
        plainNumber >= 0
    ){

        return Math.ceil(
            plainNumber * 1000
        );

    }


    /*
     * 250ms
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
     * 11.19s
     */


    const secondsMatch =

        text.match(
            /^([\d.]+)\s*s$/
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
     * 1.5m
     */


    const minutesMatch =

        text.match(
            /^([\d.]+)\s*m$/
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
     * seconds / duration
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
     * HTTP-date
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
 * GROQ RATE-LIMIT RESET HEADERS
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

            getHeader(
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
     * Берём максимальное ограничение.
     */


    return Math.max(
        ...delays
    );

}


/*
 * =========================================================
 * RETRY-AFTER FROM MESSAGE
 * =========================================================
 *
 * Groq может вернуть время только текстом:
 *
 * "Please try again in 11.19s"
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
     * seconds
     */


    const secondsMatch =

        message.match(
            /try\s+again\s+in\s+([\d.]+)\s*(?:s|sec|secs|second|seconds)\b/i
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
     * minutes
     */


    const minutesMatch =

        message.match(
            /try\s+again\s+in\s+([\d.]+)\s*(?:m|min|mins|minute|minutes)\b/i
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
 * RETRY-AFTER
 * =========================================================
 */


function getRetryAfterMilliseconds(
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


    /*
     * Используем самое консервативное
     * ограничение.
     */


    return Math.max(
        ...values
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


    if(
        getErrorCode(
            error
        )
        ===
        "rate_limit_exceeded"
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
 * LONG RATE LIMIT
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


    /*
     * Не держим один пользовательский
     * запрос слишком долго.
     */


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
 * RETRYABLE ERROR
 * =========================================================
 */


export function isRetryableAIError(
    error
) {

    /*
     * Известный model protocol glitch.
     *
     * Проверяем до x-should-retry:
     * HTTP 400 в данном конкретном случае
     * не означает некорректный Jessica request.
     */


    if(
        isTransientModelProtocolError(
            error
        )
    ){

        return true;

    }


    /*
     * Уважаем явный запрет provider.
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
    ){

        return true;

    }


    /*
     * Temporary provider failures.
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
     * Exponential backoff:
     *
     * attempt 1 → 1s
     * attempt 2 → 2s
     * attempt 3 → 4s
     */


    const multiplier =

        Math.max(
            0,
            Number(attempt || 1) - 1
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


            const canRetry =

                retryable
                &&
                attempt < maxAttempts;


            const delay =

                canRetry

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
                !canRetry
            ){

                throw error;

            }


            /*
             * =================================================
             
