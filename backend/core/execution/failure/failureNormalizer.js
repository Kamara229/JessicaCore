/*
 * =========================================================
 * JESSICA EXECUTION
 * FAILURE NORMALIZER v1
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







/*
 * =========================================================
 * NORMALIZE FAILURE
 * =========================================================
 */


export function normalizeFailure(

    failure

) {


    /*
     * =====================================================
     * EMPTY FAILURE
     * =====================================================
     */


    if (

        !failure ||

        typeof failure !== "object"

    ) {


        return {


            stage:

                "execution",



            failureType:

                "unknown",



            reason:

                "Неизвестная ошибка",



            validation:

                null,



            needsClarification:

                false,



            noVerifiedResult:

                false


        };

    }









    /*
     * =====================================================
     * STANDARD FAILURE
     * =====================================================
     */


    return {


        stage:

            failure.stage ||

            "execution",



        failureType:

            failure.failureType ||

            "execution-error",



        reason:

            failure.reason ||

            failure.text ||

            "Ошибка выполнения",



        validation:

            failure.validation ||

            null,



        needsClarification:

            failure.needsClarification === true,



        noVerifiedResult:

            failure.noVerifiedResult === true


    };

}
