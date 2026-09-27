/*
 * =========================================================
 * JESSICA EXECUTION
 * FAILURE NORMALIZER v3
 * =========================================================
 *
 * Нормализация Execution Failure.
 *
 *
 * Flow:
 *
 * Raw Failure
 *      ↓
 * Normalize
 *      ↓
 * Standard Failure Object
 *
 *
 * НЕ:
 *
 * - классифицирует ошибку;
 * - принимает решение;
 * - делает Retry;
 * - делает Replan.
 *
 * =========================================================
 */







function safeString(

    value

) {


    return String(

        value || ""

    )
    .trim();

}









export function normalizeFailure(

    failure

) {


    if(

        !failure ||

        typeof failure !== "object"

    ){


        return {


            stage:

                "execution",



            failureType:

                "unknown",



            category:

                null,



            reason:

                "Неизвестная ошибка",



            shouldRetry:

                false,



            needsClarification:

                false,



            noVerifiedResult:

                false,



            validation:

                null,



            source:

                null,



            details:

                null


        };

    }









    return {


        stage:

            safeString(

                failure.stage

            )

            ||

            "execution",







        failureType:

            safeString(

                failure.failureType

            )

            ||

            "execution-error",







        category:

            safeString(

                failure.category

            )

            ||

            null,









        reason:

            safeString(

                failure.reason

            )

            ||

            safeString(

                failure.text

            )

            ||

            "Ошибка выполнения",







        shouldRetry:

            failure.shouldRetry === true,







        needsClarification:

            failure.needsClarification === true,







        noVerifiedResult:

            failure.noVerifiedResult === true,







        validation:

            failure.validation ||

            null,







        source:

            failure.source ||

            null,







        details:

        {


            runResult:

                failure.runResult ||

                null,



            answerResult:

                failure.answerResult ||

                null,



            tool:

                failure.tool ||

                null,



            stepId:

                failure.stepId ||

                null


        }


    };

}
