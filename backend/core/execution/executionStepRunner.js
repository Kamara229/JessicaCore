/*
 * =========================================================
 * JESSICA EXECUTION STEP RUNNER v8
 * =========================================================
 *
 * Координатор одного Execution Pass.
 *
 *
 * Flow:
 *
 * Context
 *        ↓
 * Task Runner
 *        ↓
 * Answer Executor
 *        ↓
 * Validation Executor
 *        ↓
 * Completed Result
 *
 *
 * Ответственность:
 *
 * - запускать стадии Execution;
 * - сохранять статус этапов;
 * - возвращать Success / Failure.
 *
 *
 * НЕ:
 *
 * - выполняет Tools;
 * - создаёт Answer;
 * - валидирует;
 * - делает Retry;
 * - делает Replan;
 * - принимает Terminal Decision.
 *
 * =========================================================
 */



import {
    executeTaskRunner
} from "./stepExecution/taskRunnerExecutor.js";


import {
    executeAnswer
} from "./stepExecution/answerExecutor.js";


import {
    executeValidation
} from "./stepExecution/validationExecutor.js";


import {
    buildCompletedResult
} from "./executionResult.js";


import {
    registerExecutionStep
} from "./executionContext.js";









/*
 * =========================================================
 * STEP MARK
 * =========================================================
 */


function markStep(

    context,

    stage,

    status,

    data = {}

) {


    registerExecutionStep(

        context,

        {

            stage,

            status,

            ...data

        }

    );

}









/*
 * =========================================================
 * STAGE ERROR
 * =========================================================
 */


function buildStageException(

    stage,

    error

) {


    return {


        success:false,


        failure:

        {


            stage,


            failureType:

                `${stage}-exception`,



            reason:

                error?.message ||

                `Ошибка стадии ${stage}`


        }


    };

}









/*
 * =========================================================
 * EXECUTE STAGE
 * =========================================================
 */


async function executeStage(

    context,

    stage,

    executor

) {


    markStep(

        context,

        stage,

        "RUNNING"

    );





    let result;



    try {


        result =

            await executor(

                context

            );


    }

    catch(error){


        result =

            buildStageException(

                stage,

                error

            );


    }









    if (

        result?.success !== true

    ) {


        markStep(

            context,

            stage,

            "FAILED",

            {

                failure:

                    result?.failure

            }

        );



        return result;

    }









    markStep(

        context,

        stage,

        "COMPLETED"

    );









    return result;

}









/*
 * =========================================================
 * EXECUTE STEP
 * =========================================================
 */


export async function executeExecutionStep(

    context

) {


    /*
     * =====================================================
     * TASK RUNNER
     * =====================================================
     */


    const runner =

        await executeStage(

            context,

            "runner",

            executeTaskRunner

        );



    if (

        runner.success !== true

    ) {


        return runner;

    }









    /*
     * =====================================================
     * ANSWER
     * =====================================================
     */


    const answer =

        await executeStage(

            context,

            "composer",

            executeAnswer

        );



    if (

        answer.success !== true

    ) {


        return answer;

    }









    /*
     * =====================================================
     * VALIDATION
     * =====================================================
     */


    const validation =

        await executeStage(

            context,

            "validator",

            executeValidation

        );



    if (

        validation.success !== true

    ) {


        return validation;

    }









    /*
     * =====================================================
     * RESULT
     * =====================================================
     */


    return {


        success:true,


        result:

            buildCompletedResult(

                context,

                context.answerResult,

                true

            )


    };


}
