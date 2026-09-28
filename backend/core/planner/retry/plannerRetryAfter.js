/*
 * =========================================================
 * JESSICA PLANNER RETRY-AFTER
 * =========================================================
 *
 * Извлекает рекомендуемую задержку повторного запроса:
 *
 * - Retry-After header;
 * - HTTP-date;
 * - текст сообщения Groq.
 *
 * Возвращает миллисекунды либо null.
 *
 * =========================================================
 */


import {
    MAX_RETRY_AFTER
} from "./plannerRetryConfig.js";


function getHeaderValue(
    headers,
    name
) {


    if (
        !headers
    ) {

        return null;

    }


    const normalizedName =
        String(name || "")
            .toLowerCase();


    if (
        typeof headers.get === "function"
    ) {


        const value =
            headers.get(
                normalizedName
            );


        if (
            value !== null &&
            value !== undefined
        ) {

            return value;

        }

    }


    if (
        typeof headers === "object"
    ) {


        for (
            const key of Object.keys(headers)
        ) {


            if (
                String(key)
                    .toLowerCase()
                ===
                normalizedName
            ) {

                return headers[key];

            }

        }

    }


    return null;

}


function parseRetryAfterHeader(
    value
) {


    if (
        value === null ||
        value === undefined
    ) {

        return null;

    }


    const raw =
        String(value)
            .trim();


    if (
        !raw
    ) {

        return null;

    }


    const seconds =
        Number(raw);


    if (
        Number.isFinite(seconds) &&
        seconds >= 0
    ) {


        return Math.round(
            seconds * 1000
        );

    }


    const timestamp =
        Date.parse(raw);


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


function parseRetryAfterMessage(
    error
) {


    const message =
        String(

            error?.message ||

            error ||

            ""

        );


    if (
        !message
    ) {

        return null;

    }


    const secondsMatch =
        message.match(
            /try\s+again\s+in\s+([\d.]+)\s*(?:s|sec|secs|second|seconds)\b/i
        );


    if (
        secondsMatch?.[1]
    ) {


        const seconds =
            Number(
                secondsMatch[1]
            );


        if (
            Number.isFinite(seconds) &&
            seconds >= 0
        ) {


            return Math.round(
                seconds * 1000
            );

        }

    }


    const millisecondsMatch =
        message.match(
            /try\s+again\s+in\s+([\d.]+)\s*(?:ms|millisecond|milliseconds)\b/i
        );


    if (
        millisecondsMatch?.[1]
    ) {


        const milliseconds =
            Number(
                millisecondsMatch[1]
            );


        if (
            Number.isFinite(milliseconds) &&
            milliseconds >= 0
        ) {

            return Math.round(
                milliseconds
            );

        }

    }


    return null;

}


/*
 * =========================================================
 * PUBLIC
 * =========================================================
 */


export function getPlannerRetryAfter(
    error
) {


    if (
        !error
    ) {

        return null;

    }


    const headerSources = [

        error?.headers,

        error?.response?.headers

    ];


    for (
        const headers of headerSources
    ) {


        const value =
            getHeaderValue(

                headers,

                "retry-after"

            );


        const delay =
            parseRetryAfterHeader(
                value
            );


        if (
            delay !== null
        ) {


            return Math.min(

                delay,

                MAX_RETRY_AFTER

            );

        }

    }


    const messageDelay =
        parseRetryAfterMessage(
            error
        );


    if (
        messageDelay !== null
    ) {


        return Math.min(

            messageDelay,

            MAX_RETRY_AFTER

        );

    }


    return null;

}
