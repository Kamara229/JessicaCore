/*
 * =========================================================
 * JESSICA AI RETRY
 * EXECUTOR v1
 * =========================================================
 *
 * Выполняет технический AI retry loop.
 *
 *
 * Flow:
 *
 * AI operation
 *      ↓
 * success → return
 *
 * error
 *      ↓
 * retryClassifier
 *      ↓
 * retryDelay
 *      ↓
 * sleep
 *      ↓
 * repeat
 *
 *
 * НЕ:
 *
 * - знает бизнес-смысл AI request;
 * - делает semantic retry;
 * - делает Execution Replan.
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
    getAIRetryDelay,
    sleep
} from "./retryDelay.js";


/*
 * =========================================================
 * MAX ATTEMPTS
 * =========================================================
 */


function resolveMaxAttempts(
    value
) {

    if(
        Number.isInteger(value)
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
             * =============================================
             * CLASSIFY
             * =============================================
             */


            const retryable =

                isRetryableAIError(
                    error
                );


            const canRetry =

                retryable
                &&
                attempt < maxAttempts;


            /*
             * =============================================
             * DELAY
             * =============================================
             */


            const delay =

                canRetry

                    ? getAIRetryDelay(
                        error,
                        attempt
                    )

                    : 0;


            /*
             * =============================================
             * LOG
             * =============================================
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

                    retryDelay:
                        delay,

                    message:

                        getAIErrorMessage(
                            error
                        )

                        ||

                        String(error)

                }

            );


            /*
             * =============================================
             * NOT RETRYABLE
             * =============================================
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
             * =============================================
             * EXHAUSTED
             * =============================================
             */


            if(
                !canRetry
            ){

                throw error;

            }


            /*
             * =============================================
             * WAIT
             * =============================================
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
