import {
    TERMINAL_OUTCOME,
    resolveTerminalOutcome
} from "./terminalOutcomePolicy.js";


import {
    buildFailureResult,
    buildNoVerifiedResult
} from "./executionResult.js";



/*
 * =========================================================
 * JESSICA EXECUTION TERMINAL
 * =========================================================
 *
 * Финальная обработка завершения цикла.
 *
 *
 * Сюда приходят ситуации:
 *
 * - закончились попытки;
 * - нет подтверждённого результата;
 * - дальнейшее выполнение невозможно.
 *
 *
 * НЕ содержит:
 *
 * - retry;
 * - replan;
 * - runner;
 * - composer;
 * - validator.
 *
 * =========================================================
 */





/*
 * =========================================================
 * BUILD TERMINAL RESULT
 * =========================================================
 */


export function buildTerminalResult(

    context,

    failure

) {



    const outcome =
        resolveTerminalOutcome(
            failure
        );



    console.log(

        "Jessica terminal outcome:",

        JSON.stringify({

            type:
                outcome?.type || null,


            resultType:
                outcome?.resultType || null,


            failureType:
                failure?.failureType || null,


            attempt:
                context?.attempt || 0

        })

    );






    /*
     * =====================================================
     * NO VERIFIED RESULT
     * =====================================================
     */


    if (

        outcome?.type ===

        TERMINAL_OUTCOME.NO_VERIFIED_RESULT

    ) {


        return buildNoVerifiedResult(

            context,

            {

                message:
                    outcome?.message ||
                    "Не удалось подтвердить результат.",


                reason:
                    outcome?.reason ||
                    failure?.reason ||
                    "",


                stage:
                    failure?.stage ||
                    "execution",


                failureType:
                    failure?.failureType ||
                    null

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
                failure?.stage ||
                "execution",


            reason:
                outcome?.reason ||
                failure?.reason ||
                "Не удалось выполнить задачу",


            failureType:
                failure?.failureType ||
                null

        }

    );


}
