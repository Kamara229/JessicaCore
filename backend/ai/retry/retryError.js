/*
 * =========================================================
 * JESSICA AI RETRY
 * ERROR ACCESSORS v1
 * =========================================================
 *
 * Нормализует технические данные AI provider error.
 *
 *
 * Используется:
 *
 * - Retry Classifier;
 * - Retry-After parser;
 * - Retry Executor logging.
 *
 *
 * НЕ:
 *
 * - решает retry;
 * - вычисляет delay;
 * - выполняет sleep.
 *
 * =========================================================
 */


/*
 * =========================================================
 * SAFE STRING
 * =========================================================
 */


export function safeRetryString(
    value
) {

    return typeof value === "string"

        ? value.trim()

        : "";

}


/*
 * =========================================================
 * MESSAGE
 * =========================================================
 */


export function getAIErrorMessage(
    error
) {

    return safeRetryString(

        error?.message

        ||

        error?.error?.message

        ||

        error?.response
            ?.data
            ?.error
            ?.message

        ||

        error?.response
            ?.data
            ?.message

    );

}


/*
 * =========================================================
 * STATUS
 * =========================================================
 */


export function getAIErrorStatus(
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

            Number(
                candidate
            );


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
 * CODE
 * =========================================================
 */


export function getAIErrorCode(
    error
) {

    return safeRetryString(

        error?.code

        ||

        error?.error?.code

    )
        .toLowerCase();

}


/*
 * =========================================================
 * NETWORK CODE
 * =========================================================
 */


export function getAINetworkErrorCode(
    error
) {

    return safeRetryString(
        error?.code
    )
        .toUpperCase();

}


/*
 * =========================================================
 * HEADER
 * =========================================================
 */


export function getAIErrorHeader(
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

            String(
                name || ""
            )
                .trim()
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
