/*
 * =========================================================
 * JESSICA EXECUTION
 * TASK RUNNER EXECUTOR v1
 * =========================================================
 *
 * Запуск выполнения текущего Execution Plan.
 *
 *
 * Flow:
 *
 * Execution Context
 *        ↓
 * Task Runner
 *        ↓
 * Run Result
 *
 *
 * НЕ:
 *
 * - создаёт Plan;
 * - делает Retry;
 * - делает Replan;
 * - создаёт Answer;
 * - выполняет Validation.
 *
 * =========================================================
 */


import {
    runPlan
} from "../../taskRunner.js";









/*
 * =========================================================
 * FAILURE
 * =========================================================
 */


function buildFailure(

    stage,

    failureType,

    reason,

    extra = {}

) {


    return {


        success:false,


        failure:

        {


            stage,


            failureType,


            reason,


            ...extra


        }


    };

}









/*
 * =========================================================
 * EXECUTE TASK RUNNER
 * =========================================================
 */


export async function executeTaskRunner(

    context

) {


    if (
        !context
    ) {


        return buildFailure(

            "runner",

            "missing-context",

            "Execution context отсутствует"

        );

    }









    try {


        context.runResult =

            await runPlan(

                context.plan,

                context.task

            );



    }

    catch(error){


        return buildFailure(

            "runner",

            "runner-exception",

            error?.message ||

            "Ошибка выполнения TaskRunner"

        );


    }









    if (

        context.runResult?.success !== true

    ) {



        return buildFailure(

            context.runResult?.stage ||

            "runner",


            context.runResult?.failureType ||

            "runner-failure",


            context.runResult?.reason ||

            "План не выполнен",


            {

                runResult:

                    context.runResult

            }

        );


    }









    return {


        success:true,


        runResult:

            context.runResult


    };


}
