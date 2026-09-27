/*
 * =========================================================
 * JESSICA EXECUTION TERMINAL v6
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
 * Execution Result
 *
 *
 * Ответственность:
 *
 * - выбрать финальный Result Builder;
 * - сформировать terminal outcome.
 *
 *
 * НЕ:
 *
 * - анализирует ошибки;
 * - делает Retry;
 * - делает Replan;
 * - вызывает Planner;
 * - вызывает Tools;
 * - изменяет Context.
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
 * SAFE STRING
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

        ||

        failure.type === "needs-clarification"

    ) {


        terminalType =

            TERMINAL_TYPE.CLARIFICATION;


    }









    if (

        failure.noVerifiedResult === true

        ||

        failureType === "no_verified_result"

        ||

        failureType === "no-verified-result"

    ) {


        terminalType =

            TERMINAL_TYPE.NO_VERIFIED_RESULT;


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

                context?.attempt ||

                0,



            type:

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

        normalized.terminalType ===

        TERMINAL_TYPE.CLARIFICATION

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

        normalized.terminalType ===

        TERMINAL_TYPE.NO_VERIFIED_RESULT

    ) {


        return buildNoVerifiedResult(

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
     * =====================================================
     * FAILURE
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
