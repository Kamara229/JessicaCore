/*
 * =========================================================
 * JESSICA AI RETRY
 * EXECUTOR v3
 * =========================================================
 *
 * Выполняет technical AI retry loop.
 *
 *
 * AI operation
 *      ↓
 * success
 *      → return
 *
 * error
 *      ↓
 * classify
 *      ↓
 * retry-after
 *      ↓
 * delay
 *      ↓
 * retry
 *
 * =========================================================
 */


import {
    MAX_AI_ATTEMPTS
} from "./retryConstants.js";


import {
    getAIErrorCode,
    getAIErrorMessage,
    getAIErrorStatus,
    safeRetryString
} from "./retryError.js";


import {
    isRetryableAIError
} from "./retryClassifier.js";


import {
    getRetryAfterDetails
} from "./retryAfter.js";


import {
    getAIRetryDelay,
    sleep
} from "./retryDelay.js";


import {
    registerAIProviderLimit,
    throwIfAIProviderCoolingDown
} from "../providerAvailability.js";


/*
 * =========================================================
 * MAX ATTEMPTS
 * =========================================================
 */


function resolveMaxAttempts(
    value
) {

    if(
        Number.isInteger(
            value
        )
        &&
        value > 0
    ){

        return value;

    }


    return MAX_AI_ATTEMPTS;

}


/*
 * =========================================================
 * LABEL
 * =========================================================
 */


function resolveLabel(
    value
) {

    return (

        safeRetryString(
            value
        )

        ||

        "AI request"

    );

}


/*
 * =========================================================
 * EXECUTE
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

        resolveLabel(
            options.label
        );


    const maxAttempts =

        resolveMaxAttempts(
            options.maxAttempts
        );


    let lastError =
        null;


    /*
     * Provider circuit breaker.
     *
     * Если предыдущий AI request уже получил
     * long/daily quota cooldown, новый сетевой
     * вызов не отправляем.
     */


    throwIfAIProviderCoolingDown(
        label
    );


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


            /*
             * =================================================
             * CLASSIFY
             * =================================================
             */


            const retryable =

                isRetryableAIError(
                    error
                );


            /*
             * Long/daily quota открывает общий cooldown.
             *
             * Short TPM/RPM retry остаётся локальным
             * и circuit breaker не включает.
             */


            if(
                !retryable
                &&
                getAIErrorStatus(
                    error
                ) === 429
            ){

                registerAIProviderLimit(
                    error
                );

            }


            const canRetry =

                retryable
                &&
                attempt < maxAttempts;


            /*
             * =================================================
             * PROVIDER RETRY INFO
             * =================================================
             */


            const retryAfter =

                getRetryAfterDetails(
                    error
                );


            /*
             * =================================================
             * DELAY
             * =================================================
             */


            const delay =

                canRetry

                    ? getAIRetryDelay(
                        error,
                        attempt
                    )

                    : 0;


            /*
             * =================================================
             * LOG
             * =================================================
             */


            console.warn(

                `${label} failed [${attempt}/${maxAttempts}]:`,

                {

                    status:

                        getAIErrorStatus(
                            error
                        ),


                    code:

                        getAIErrorCode(
                            error
                        )

                        ||

                        null,


                    retryable,


                    providerRetryAfterMs:

                        retryAfter
                            ?.milliseconds

                        ??

                        null,


                    retryAfterSource:

                        retryAfter
                            ?.source

                        ||

                        null,


                    retryDelay:
                        delay,


                    message:

                        getAIErrorMessage(
                            error
                        )

                        ||

                        String(
                            error
                        )

                }

            );


            /*
             * =================================================
             * NOT RETRYABLE
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
             * EXHAUSTED
             * =================================================
             */


            if(
                !canRetry
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
