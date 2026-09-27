/*
 * =========================================================
 * JESSICA EXECUTION
 * TASK RUNNER EXECUTOR v2
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
 * Ответственность:
 *
 * - передать Plan в TaskRunner;
 * - сохранить Run Result;
 * - нормализовать ошибку выполнения.
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
 * NORMALIZE RUN FAILURE
 * =========================================================
 */


function normalizeRunFailure(

    runResult

) {


    return {


        stage:

            runResult?.stage ||

            "runner",



        failureType:

            runResult?.failureType ||

            "runner-failure",



        reason:

            runResult?.reason ||

            runResult?.text ||

            "План не выполнен",



        shouldRetry:

            runResult?.shouldRetry === true,



        needsClarification:

            runResult?.needsClarification === true,



        runResult

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


        return {


            success:false,


            failure:

                normalizeRunFailure(

                    context.runResult

                )


        };


    }









    return {


        success:true,


        runResult:

            context.runResult


    };


}
