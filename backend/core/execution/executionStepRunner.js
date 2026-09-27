/*
 * =========================================================
 * JESSICA EXECUTION STEP RUNNER v7
 * =========================================================
 *
 * Центральный координатор одного Execution Pass.
 *
 *
 * Flow:
 *
 * Execution Context
 *        ↓
 * Task Runner Executor
 *        ↓
 * Answer Executor
 *        ↓
 * Validation Executor
 *        ↓
 * Execution Result
 *
 *
 * Ответственность:
 *
 * - запускать стадии Execution;
 * - передавать Context;
 * - возвращать Success / Failure.
 *
 *
 * НЕ:
 *
 * - выполняет Tools;
 * - создаёт Answer;
 * - валидирует данные;
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


    markStep(

        context,

        "runner",

        "RUNNING"

    );



    const runner =

        await executeTaskRunner(

            context

        );



    if (

        runner.success !== true

    ) {


        markStep(

            context,

            "runner",

            "FAILED",

            {

                failure:

                    runner.failure

            }

        );



        return runner;

    }



    markStep(

        context,

        "runner",

        "COMPLETED"

    );









    /*
     * =====================================================
     * ANSWER
     * =====================================================
     */


    markStep(

        context,

        "composer",

        "RUNNING"

    );



    const answer =

        await executeAnswer(

            context

        );



    if (

        answer.success !== true

    ) {


        markStep(

            context,

            "composer",

            "FAILED",

            {

                failure:

                    answer.failure

            }

        );



        return answer;

    }



    markStep(

        context,

        "composer",

        "COMPLETED"

    );









    /*
     * =====================================================
     * VALIDATION
     * =====================================================
     */


    markStep(

        context,

        "validator",

        "RUNNING"

    );



    const validation =

        await executeValidation(

            context

        );



    if (

        validation.success !== true

    ) {


        markStep(

            context,

            "validator",

            "FAILED",

            {

                failure:

                    validation.failure

            }

        );



        return validation;

    }



    markStep(

        context,

        "validator",

        "COMPLETED"

    );









    /*
     * =====================================================
     * COMPLETED RESULT
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
