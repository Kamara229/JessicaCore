/*
 * =========================================================
 * JESSICA TASK RUNNER
 * =========================================================
 *
 * Центральный координатор выполнения плана.
 *
 *
 * Flow:
 *
 * Plan
 *   ↓
 * Runtime Validation
 *   ↓
 * Plan Execution Loop
 *   ↓
 * Step Execution
 *   ↓
 * Result
 *
 *
 * НЕ:
 *
 * - строит планы;
 * - вызывает Planner;
 * - делает Replan;
 * - выполняет Retry;
 * - валидирует финальный ответ;
 * - работает с Experience.
 *
 * =========================================================
 */


import {
    validatePlanForExecution
} from "./taskRunner/planRuntimeValidator.js";


import {
    executePlanSteps
} from "./taskRunner/planExecutionLoop.js";


import {
    buildFailure,
    buildSuccess,
    buildInvalidPlanFailure
} from "./taskRunner/resultHandler.js";









/*
 * =========================================================
 * RUN PLAN
 * =========================================================
 */


export async function runPlan(

    plan,

    task = ""

) {



    /*
     * =====================================================
     * PLAN VALIDATION
     * =====================================================
     */


    const validation =

        validatePlanForExecution(

            plan

        );







    if (

        validation.success !== true

    ) {


        return buildInvalidPlanFailure(

            validation

        );


    }









    /*
     * =====================================================
     * NO TOOLS
     * =====================================================
     */


    if (

        validation.noTools === true

    ) {


        return buildSuccess(

            []

        );


    }









    /*
     * =====================================================
     * EXECUTION LOOP
     * =====================================================
     */


    try {


        const result =

            await executePlanSteps({

                plan,

                task,

                results: []

            });








        if (

            result?.success !== true

        ) {


            return result;


        }








        return buildSuccess(

            result.results || []

        );



    }

    catch(error){


        console.error(

            "Jessica TaskRunner execution error:",

            error

        );



        return buildFailure({

            stage:
                "runner",


            failureType:
                "runner-exception",


            text:
                error?.message ||

                "Ошибка выполнения плана",


            results:
                []

        });


    }



}
