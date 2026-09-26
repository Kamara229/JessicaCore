/*
 * =========================================================
 * JESSICA EXECUTION TERMINAL v3
 * =========================================================
 *
 * Финальный слой завершения Execution Cycle.
 *
 *
 * Получает:
 *
 * Execution Context
 * Failure Decision
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
 * - решает retry;
 * - решает replan;
 * - вызывает Planner;
 * - вызывает Tools;
 * - меняет Experience.
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
 * LOG TERMINAL
 * =========================================================
 */


function logTerminal(
    context,
    decision
) {


    console.log(

        "Jessica terminal:",

        JSON.stringify({

            status:
                decision?.status ||
                null,


            reason:
                decision?.reason ||
                "",


            stage:
                decision?.stage ||
                "",


            attempt:
                context?.attempt || 0


        })

    );


}









/*
 * =========================================================
 * BUILD TERMINAL RESULT
 * =========================================================
 *
 * Главная точка выхода.
 *
 * =========================================================
 */


export function buildTerminalResult(

    context,

    decision = {}

) {


    logTerminal(

        context,

        decision

    );





    const status =

        decision.status ||
        "FAILED";








    /*
     * =====================================================
     * NEEDS CLARIFICATION
     * =====================================================
     */


    if (

        status ===
        "NEEDS_CLARIFICATION"

    ) {


        return buildClarificationResult(

            context,

            {

                stage:

                    decision.stage ||
                    "execution",



                reason:

                    normalizeString(
                        decision.reason
                    )

                    ||

                    "Требуется уточнение"


            }

        );


    }









    /*
     * =====================================================
     * NO VERIFIED RESULT
     * =====================================================
     */


    if (

        status ===
        "NO_VERIFIED_RESULT"

    ) {


        return buildNoVerifiedResult(

            context,

            normalizeString(

                decision.reason

            )

            ||

            "Результат не удалось подтвердить."

        );


    }









    /*
     * =====================================================
     * FAILED
     * =====================================================
     */


    return buildFailureResult(

        context,

        {

            stage:

                decision.stage ||
                "execution",



            reason:

                normalizeString(
                    decision.reason
                )

                ||

                "Не удалось выполнить задачу",



            failureType:

                decision.failureType ||
                "execution-failure"


        }

    );


}
