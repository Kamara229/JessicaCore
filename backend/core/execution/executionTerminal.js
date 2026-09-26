/*
 * =========================================================
 * JESSICA EXECUTION TERMINAL v4
 * =========================================================
 *
 * Финальный слой завершения Execution Cycle.
 *
 *
 * Получает:
 *
 * Execution Context
 * Failure
 *
 *
 * Возвращает:
 *
 * Terminal Execution Result
 *
 *
 * Ответственность:
 *
 * - сформировать финальный результат;
 * - выбрать правильный Result Builder.
 *
 *
 * НЕ:
 *
 * - анализирует ошибки;
 * - делает Retry;
 * - делает Replan;
 * - вызывает Planner;
 * - вызывает Tools;
 * - изменяет Experience.
 *
 * =========================================================
 */


import {
    buildFailureResult,
    buildNoVerifiedResult,
    buildClarificationResult
} from "./executionResult.js";









/*
 * =========================================================
 * NORMALIZE STRING
 * =========================================================
 */


function normalizeString(
    value
) {


    return String(
        value || ""
    )
    .trim();


}









/*
 * =========================================================
 * NORMALIZE FAILURE
 * =========================================================
 */


function normalizeFailure(
    failure
) {


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
                "Неизвестная ошибка"



        };

    }






    return {


        stage:

            failure.stage ||

            "execution",




        failureType:

            failure.failureType ||

            "execution-failure",




        reason:

            normalizeString(
                failure.reason
            )
            ||

            "Не удалось выполнить задачу",




        needsClarification:

            failure.needsClarification === true,




        noVerifiedResult:

            failure.noVerifiedResult === true



    };


}









/*
 * =========================================================
 * LOG
 * =========================================================
 */


function logTerminal(
    context,
    failure
) {


    console.log(

        "Jessica Terminal Result:",

        {

            executionId:

                context?.executionId ||
                null,


            attempt:

                context?.attempt ||
                0,


            stage:

                failure.stage,


            failureType:

                failure.failureType

        }

    );


}









/*
 * =========================================================
 * BUILD TERMINAL RESULT
 * =========================================================
 */


export function buildTerminalResult(

    context,

    failure = {}

) {



    const normalized =

        normalizeFailure(
            failure
        );





    logTerminal(

        context,

        normalized

    );









    /*
     * =====================================================
     * CLARIFICATION
     * =====================================================
     */


    if (
        normalized.needsClarification === true
    ) {


        return buildClarificationResult(

            context,

            {

                stage:

                    normalized.stage,



                reason:

                    normalized.reason


            }

        );


    }









    /*
     * =====================================================
     * NO VERIFIED RESULT
     * =====================================================
     */


    if (
        normalized.noVerifiedResult === true
    ) {


        return buildNoVerifiedResult(

            context,

            {

                message:

                    normalized.reason,



                reason:

                    normalized.reason,



                stage:

                    normalized.stage,



                failureType:

                    normalized.failureType


            }

        );


    }









    /*
     * =====================================================
     * FINAL FAILURE
     * =====================================================
     */


    return buildFailureResult(

        context,

        {

            stage:

                normalized.stage,



            reason:

                normalized.reason,



            failureType:

                normalized.failureType


        }

    );


}









/*
 * =========================================================
 * HELPERS
 * =========================================================
 */


export function isTerminalFailure(
    failure
) {


    return Boolean(

        failure &&
        typeof failure === "object"

    );

}
