/*
 * =========================================================
 * JESSICA EXECUTION STEP RUNNER v10
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
    executeAnswerValidation
} from "./stepExecution/answerValidationCoordinator.js";


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

    executor,

    options = {}

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


        let failureStatus =
            "FAILED";


        if(
            typeof options?.failureStatusResolver === "function"
        ){

            const resolvedStatus =
                options.failureStatusResolver(
                    result
                );


            if(
                typeof resolvedStatus === "string"
                &&
                resolvedStatus.trim()
            ){

                failureStatus =
                    resolvedStatus.trim();

            }

        }


        markStep(

            context,

            stage,

            failureStatus,

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
     * ANSWER + VALIDATION
     * =====================================================
     *
     * Presentation semantic repair выполняется локально
     * и не повторяет Task Runner / Tools / Plan.
     * =====================================================
     */


    const answerValidation =

        await executeAnswerValidation(

            context,

            executeStage

        );


    if (

        answerValidation.success !== true

    ) {


        return answerValidation;

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

                context.validationResult?.degraded !== true &&
                context.validationResult?.outcomeType !== "no_verified_result"

            )


    };


}
