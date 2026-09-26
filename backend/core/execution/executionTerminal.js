/*
 * =========================================================
 * JESSICA EXECUTION TERMINAL v5
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
 * НЕ:
 *
 * - анализирует ошибки;
 * - делает Retry;
 * - делает Replan;
 * - вызывает Planner;
 * - вызывает Tools.
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
 * TERMINAL TYPES
 * =========================================================
 */


export const TERMINAL_TYPE = {


    CLARIFICATION:
        "CLARIFICATION",


    NO_VERIFIED_RESULT:
        "NO_VERIFIED_RESULT",


    FAILURE:
        "FAILURE"


};









/*
 * =========================================================
 * STRING
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
                "Неизвестная ошибка",



            terminalType:
                TERMINAL_TYPE.FAILURE


        };

    }






    const failureType =

        safeString(
            failure.failureType
        )
        ||
        "execution-failure";







    let terminalType =

        TERMINAL_TYPE.FAILURE;







    if (

        failure.needsClarification === true

    ) {


        terminalType =

            TERMINAL_TYPE.CLARIFICATION;


    }







    if (

        failure.noVerifiedResult === true

        ||

        failureType ===
        "no-verified-result"

        ||

        failureType ===
        "no_verified_result"

    ) {


        terminalType =

            TERMINAL_TYPE
                .NO_VERIFIED_RESULT;


    }







    return {


        stage:

            safeString(
                failure.stage
            )
            ||
            "execution",




        failureType,



        reason:

            safeString(
                failure.reason
            )
            ||
            "Не удалось выполнить задачу",



        terminalType


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

        "Jessica Terminal:",

        {


            executionId:

                context?.executionId ||
                null,



            attempt:

                context?.attempt || 0,



            terminalType:

                failure.terminalType,



            stage:

                failure.stage,



            failureType:

                failure.failureType


        }

    );


}









/*
 * =========================================================
 * ATTACH TERMINAL META
 * =========================================================
 */


function attachTerminalMeta(

    result,

    failure

) {


    return {


        ...result,


        terminal:

        {


            type:

                failure.terminalType,



            stage:

                failure.stage,



            failureType:

                failure.failureType


        }


    };

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
     * USER CLARIFICATION
     * =====================================================
     */


    if (

        normalized.terminalType ===
        TERMINAL_TYPE.CLARIFICATION

    ) {


        return attachTerminalMeta(

            buildClarificationResult(

                context,

                {


                    stage:

                        normalized.stage,



                    reason:

                        normalized.reason


                }

            ),


            normalized

        );


    }









    /*
     * =====================================================
     * NO VERIFIED RESULT
     * =====================================================
     */


    if (

        normalized.terminalType ===
        TERMINAL_TYPE.NO_VERIFIED_RESULT

    ) {


        return attachTerminalMeta(

            buildNoVerifiedResult(

                context,

                {


                    stage:

                        normalized.stage,



                    reason:

                        normalized.reason,



                    failureType:

                        normalized.failureType


                }

            ),


            normalized

        );


    }









    /*
     * =====================================================
     * FINAL FAILURE
     * =====================================================
     */


    return attachTerminalMeta(

        buildFailureResult(

            context,

            {


                stage:

                    normalized.stage,



                reason:

                    normalized.reason,



                failureType:

                    normalized.failureType


            }

        ),


        normalized

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
