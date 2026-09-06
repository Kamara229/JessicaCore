/*
 * =========================================================
 * JESSICA AI RETRY
 * =========================================================
 *
 * Общая техническая политика повторов AI-запросов.
 *
 * Используется для временных ошибок:
 *
 * - 429 rate limit;
 * - 408 timeout;
 * - 5xx ошибки провайдера;
 * - временные сетевые сбои.
 *
 * Это НЕ semantic retry.
 *
 * Semantic retry:
 * Validator → Replanner.
 *
 * Technical retry:
 * AI provider error → wait → repeat same request.
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


const MAX_RETRY_DELAY_MS =
    10000;


/*
 * =========================================================
 * SLEEP
 * =========================================================
 */


export function sleep(
    milliseconds
) {

    const delay =
        Number.isFinite(milliseconds)
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


    return Number.isFinite(status)
        ? status
        : 0;

}


/*
 * =========================================================
 * RETRYABLE ERROR
 * =========================================================
 */


export function isRetryableAIError(
    error
) {

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
     * Некоторые сетевые ошибки
     * могут приходить без HTTP status.
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
 * RETRY-AFTER
 * =========================================================
 */


function getRetryAfterMilliseconds(
    error
) {

    const headers =
        error?.headers;


    if (!headers) {

        return null;

    }


    let value =
        null;


    try {

        if (
            typeof headers.get === "function"
        ) {

            value =
                headers.get(
                    "retry-after"
                );

        } else {

            value =
                headers["retry-after"] ??
                headers["Retry-After"];

        }

    } catch {

        value =
            null;

    }


    if (
        value === null ||
        value === undefined ||
        value === ""
    ) {

        return null;

    }


    /*
     * Обычно Retry-After приходит
     * в секундах.
     */


    const seconds =
        Number(
            value
        );


    if (
        Number.isFinite(seconds) &&
        seconds >= 0
    ) {

        return Math.ceil(
            seconds * 1000
        );

    }


    /*
     * Стандарт HTTP также допускает дату.
     */


    const timestamp =
        Date.parse(
            String(value)
        );


    if (
        Number.isFinite(timestamp)
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

        /*
         * Добавляем небольшой запас,
         * чтобы не повторить запрос
         * ровно на границе rate limit.
         */
        return Math.min(
            retryAfter + 250,
            MAX_RETRY_DELAY_MS
        );

    }


    /*
     * Exponential backoff:
     *
     * attempt 1 → 1 сек
     * attempt 2 → 2 сек
     * attempt 3 → 4 сек
     */


    const multiplier =
        Math.max(
            0,
            attempt - 1
        );


    const delay =
        DEFAULT_RETRY_DELAY_MS *
        Math.pow(
            2,
            multiplier
        );


    return Math.min(
        delay,
        MAX_RETRY_DELAY_MS
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


            if (
                !retryable ||
                attempt >= maxAttempts
            ) {

                throw error;

            }


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
