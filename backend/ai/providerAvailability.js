/*
 * =========================================================
 * JESSICA AI PROVIDER AVAILABILITY v1
 * =========================================================
 *
 * In-process circuit breaker для Groq quota.
 *
 *
 * Задача:
 *
 * если provider уже сообщил, что следующий
 * запрос бессмысленен в ближайшие минуты,
 * Jessica не должна продолжать отправлять
 * одинаковые AI requests и тратить latency.
 *
 *
 * Это НЕ отключает Groq навсегда.
 * После cooldown provider снова доступен.
 *
 * =========================================================
 */


import {
    getAIErrorStatus,
    getAIErrorMessage
} from "./retry/retryError.js";


import {
    getRetryAfterDetails
} from "./retry/retryAfter.js";


import {
    MAX_IMMEDIATE_RETRY_WAIT_MS
} from "./retry/retryConstants.js";


const DEFAULT_COOLDOWN_MS =
    60_000;


const MAX_COOLDOWN_MS =
    15 * 60_000;


let state = {

    blockedUntil:
        0,

    reason:
        "",

    source:
        null

};


/*
 * =========================================================
 * HELPERS
 * =========================================================
 */


function now()
{

    return Date.now();

}


function normalizeMessage(
    error
) {

    return String(

        getAIErrorMessage(
            error
        )

        ||

        ""

    )
        .trim()
        .toLowerCase();

}


function isDailyQuota(
    error
) {

    const message =

        normalizeMessage(
            error
        );


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


function resetIfExpired()
{

    if(
        state.blockedUntil > 0
        &&
        state.blockedUntil <= now()
    ){

        state = {

            blockedUntil:
                0,

            reason:
                "",

            source:
                null

        };

    }

}


/*
 * =========================================================
 * PUBLIC STATE
 * =========================================================
 */


export function getAIProviderAvailability()
{

    resetIfExpired();


    const remainingMs =

        Math.max(

            0,

            state.blockedUntil - now()

        );


    return {

        available:
            remainingMs === 0,

        blocked:
            remainingMs > 0,

        remainingMs,

        blockedUntil:

            remainingMs > 0

                ? new Date(
                    state.blockedUntil
                )
                    .toISOString()

                : null,

        reason:
            state.reason || null,

        source:
            state.source

    };

}


/*
 * =========================================================
 * REGISTER LIMIT
 * =========================================================
 */


export function registerAIProviderLimit(
    error
) {

    if(
        getAIErrorStatus(
            error
        ) !== 429
    ){

        return getAIProviderAvailability();

    }

    // Provider rejects the requested output size regardless of waiting.
    // This is a request-shape error, not a quota cooldown.
    const detail=normalizeMessage(error);
    if (detail.includes("request too large") ||
        detail.includes("expected output tokens exceed") ||
        detail.includes("reduce max_tokens")) {
        return getAIProviderAvailability();
    }


    const retryAfter =

        getRetryAfterDetails(
            error
        );


    const retryAfterMs =

        Number(
            retryAfter?.milliseconds
        );


    const shouldOpenCircuit =

        isDailyQuota(
            error
        )

        ||

        (
            Number.isFinite(
                retryAfterMs
            )

            &&

            retryAfterMs
            >
            MAX_IMMEDIATE_RETRY_WAIT_MS
        );


    if(
        !shouldOpenCircuit
    ){

        return getAIProviderAvailability();

    }


    const cooldown =

        Math.min(

            Math.max(

                Number.isFinite(
                    retryAfterMs
                )

                    ? retryAfterMs

                    : DEFAULT_COOLDOWN_MS,

                DEFAULT_COOLDOWN_MS

            ),

            MAX_COOLDOWN_MS

        );


    const blockedUntil =

        now()
        +
        cooldown;


    if(
        blockedUntil
        >
        state.blockedUntil
    ){

        state = {

            blockedUntil,

            reason:

                isDailyQuota(
                    error
                )

                    ? "daily-quota"

                    : "long-rate-limit",

            source:

                retryAfter?.source

                ||

                null

        };


        console.warn(

            "Jessica AI provider cooldown opened:",

            {

                reason:
                    state.reason,

                cooldownMs:
                    cooldown,

                blockedUntil:
                    new Date(
                        blockedUntil
                    )
                        .toISOString(),

                retryAfterSource:
                    state.source

            }

        );

    }


    return getAIProviderAvailability();

}


/*
 * =========================================================
 * FAST FAIL
 * =========================================================
 */


export function throwIfAIProviderCoolingDown(
    label = "AI request"
) {

    const availability =

        getAIProviderAvailability();


    if(
        availability.available
    ){

        return;

    }


    const error =

        new Error(

            (
                label
                +
                " skipped: AI provider is temporarily unavailable"
            )

        );


    error.code =
        "AI_PROVIDER_COOLDOWN";


    error.status =
        429;


    error.retryAfterMs =
        availability.remainingMs;


    error.providerCooldown =
        true;


    throw error;

}
