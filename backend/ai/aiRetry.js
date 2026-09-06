/*
 * =========================================================
 * JESSICA AI RETRY
 * =========================================================
 *
 * Общая техническая политика повторов AI-запросов.
 *
 * Повторяем только действительно временные ошибки.
 *
 * ВАЖНО:
 *
 * - TPM rate limit с коротким Retry-After → retry;
 * - timeout / 5xx → retry;
 * - дневная квота / долгий Retry-After → НЕ retry;
 * - x-should-retry: false → НЕ retry.
 *
 * Semantic retry здесь НЕ выполняется.
 * Semantic retry = Validator → Replanner.
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


/*
 * Не держим пользовательский запрос
 * десятки минут из-за квоты провайдера.
 */
const MAX_IMMEDIATE_RETRY_WAIT_MS =
    30_000;


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
 * HEADER
 * =========================================================
 */


function getHeader(
    error,
    name
) {

    const headers =
        error?.headers;


    if (!headers) {

        return null;

    }


    try {

        if (
            typeof headers.get === "function"
        ) {

            return headers.get(
                name
            );

        }


        return (
            headers[name] ??
            headers[name.toLowerCase()] ??
            null
        );

    } catch {

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

    const status =
        Number(
            error?.status
        );


    return Number.isFinite(
        status
    )
        ? status
        : 0;

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
        String(
            getHeader(
                error,
                "x-should-retry"
            ) || ""
        )
            .trim()
            .toLowerCase();


    /*
     * Провайдер явно запретил повтор.
     */
    if (
        value === "false"
    ) {

        return false;

    }


    return true;

}


/*
 * =========================================================
 * RETRY-AFTER
 * =========================================================
 */


function getRetryAfterMilliseconds(
    error
) {

    const value =
        getHeader(
            error,
            "retry-after"
        );


    if (
        value === null ||
        value === undefined ||
        value === ""
    ) {

        return null;

    }


    /*
     * Retry-After в секундах.
     */


    const seconds =
        Number(
            value
        );


    if (
        Number.isFinite(
            seconds
        ) &&
        seconds >= 0
    ) {

        return Math.ceil(
            seconds * 1000
        );

    }


    /*
     * HTTP также допускает дату.
     */


    const timestamp =
        Date.parse(
            String(
                value
            )
        );


    if (
        Number.isFinite(
            timestamp
        )
    ) {

        return Math.max(
            0,
            timestamp - Date.now()
        );

    }


    return null;

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
        String(
            error?.message ||
            error?.error?.message ||
            ""
        )
            .toLowerCase();


    /*
     * Дневные / долгосрочные квоты.
     */


    if (
        message.includes(
            "tokens per day"
        ) ||
        message.includes(
            "(tpd)"
        )
    ) {

        return true;

    }


    const retryAfter =
        getRetryAfterMilliseconds(
            error
        );


    /*
     * Если провайдер просит ждать больше
     * 30 секунд, не блокируем текущую задачу.
     */


    if (
        retryAfter !== null &&
        retryAfter >
            MAX_IMMEDIATE_RETRY_WAIT_MS
    ) {

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
     * Сначала уважаем прямое указание
     * самого AI-провайдера.
     */


    if (
        !providerAllowsRetry(
            error
        )
    ) {

        return false;

    }


    const status =
        getErrorStatus(
            error
        );


    /*
     * Rate limit.
     */


    if (
        status === 429
    ) {

        /*
         * Долгосрочную квоту внутри
         * пользовательского запроса
         * повторять бессмысленно.
         */


        if (
            isLongRateLimit(
                error
            )
        ) {

            return false;

        }


        return true;

    }


    /*
     * Request timeout.
     */


    if (
        status === 408
    ) {

        return true;

    }


    /*
     * Temporary provider errors.
     */


    if (
        status >= 500 &&
        status <= 599
    ) {

        return true;

    }


    /*
     * Network errors without HTTP status.
     */


    const code =
        String(
            error?.code || ""
        )
            .trim()
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


    if (
        retryAfter !== null
    ) {

        return Math.min(
            retryAfter + 250,
            MAX_IMMEDIATE_RETRY_WAIT_MS
        );

    }


    /*
     * Exponential backoff:
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
        DEFAULT_RETRY_DELAY_MS *
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

    if (
        typeof operation !== "function"
    ) {

        throw new Error(
            "AI retry operation не является функцией"
        );

    }


    const label =
        String(
            options.label ||
            "AI request"
        );


    const maxAttempts =
        Number.isInteger(
            options.maxAttempts
        ) &&
        options.maxAttempts > 0
            ? options.maxAttempts
            : MAX_AI_ATTEMPTS;


    let lastError =
        null;


    for (
        let attempt = 1;
        attempt <= maxAttempts;
        attempt++
    ) {

        try {

            return await operation();

        } catch (error) {

            lastError =
                error;


            const retryable =
                isRetryableAIError(
                    error
                );


            console.warn(
                `${label} failed ` +
                `[${attempt}/${maxAttempts}]:`,
                error?.message ||
                error
            );


            /*
             * =================================================
             * DO NOT RETRY
             * =================================================
             */


            if (
                !retryable
            ) {

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


            if (
                attempt >=
                maxAttempts
            ) {

                throw error;

            }


            /*
             * =================================================
             * WAIT
             * =================================================
             */


            const delay =
                getAIRetryDelay(
                    error,
                    attempt
                );


            console.log(
                `${label} technical retry in ${delay}ms`
            );


            await sleep(
                delay
            );

        }

    }


    throw (
        lastError ||
        new Error(
            `${label} failed`
        )
    );

}
