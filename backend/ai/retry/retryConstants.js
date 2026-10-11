/*
 * =========================================================
 * JESSICA AI RETRY
 * CONSTANTS v1
 * =========================================================
 *
 * Только конфигурация AI technical retry.
 *
 * НЕ:
 *
 * - анализирует ошибки;
 * - вычисляет delay;
 * - выполняет retry.
 *
 * =========================================================
 */


export const MAX_AI_ATTEMPTS =
    3;


export const DEFAULT_RETRY_DELAY_MS =
    1000;


/*
 * Не держим один пользовательский запрос
 * десятки минут из-за provider quota.
 */


export const MAX_IMMEDIATE_RETRY_WAIT_MS =
    30_000;


/*
 * Небольшой запас после Retry-After.
 *
 * Например:
 *
 * provider → 11.19s
 * Jessica  → ~11.54s
 */


export const RETRY_SAFETY_MARGIN_MS =
    350;


/*
 * Network errors,
 * которые можно считать временными.
 */


export const RETRYABLE_NETWORK_CODES =
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
