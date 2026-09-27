/*
 * =========================================================
 * JESSICA TASK RUNNER
 * PLAN EXECUTION LOOP v2
 * =========================================================
 *
 * Выполнение списка шагов плана.
 *
 *
 * Flow:
 *
 * Plan Steps
 *      ↓
 * Step Iterator
 *      ↓
 * Execute Step
 *      ↓
 * Collect Results
 *
 *
 * НЕ:
 *
 * - проверяет инструменты;
 * - разрешает arguments;
 * - выполняет Tools;
 * - делает Retry;
 * - делает Replan.
 *
 * =========================================================
 */


import {
    executeStep
} from "./executeStep.js";









/*
 * =========================================================
 * BUILD CONTEXT
 * =========================================================
 */


function buildSelectionContext(

    plan,

    task

) {


    return (

        String(
            task || ""
        )
        .trim()

        ||

        [

            plan?.intent,

            plan?.reasoningSummary,

            plan?.evidence?.reason

        ]

        .filter(Boolean)

        .join("\n")

    );

}









/*
 * =========================================================
 * EXECUTION LOOP
 * =========================================================
 */


export async function executePlanSteps({

    plan,

    task,

    results = []

}) {


    if (

        !Array.isArray(
            plan?.steps
        )

    ) {


        return {


            success:false,


            stage:
                "runner",


            failureType:
                "invalid-steps",


            reason:
                "В плане отсутствуют шаги",


            results


        };

    }









    const selectionContext =

        buildSelectionContext(

            plan,

            task

        );









    for (

        let index = 0;

        index < plan.steps.length;

        index++

    ) {



        const step =

            plan.steps[index];









        const result =

            await executeStep({

                step,

                index,

                results,

                selectionContext


            });









        if (

            result?.success !== true

        ) {


            return result;


        }



    }









    return {


        success:true,


        results


    };


}
