/*
 * =========================================================
 * JESSICA EXECUTION STEP RUNNER v5
 * =========================================================
 *
 * Выполняет один проход Execution Cycle.
 *
 *
 * Flow:
 *
 * Execution Context
 *        ↓
 * Task Runner
 *        ↓
 * Answer Composer
 *        ↓
 * Validator
 *        ↓
 * Execution Result
 *
 *
 * НЕ:
 *
 * - retry;
 * - replan;
 * - terminal decision;
 * - Learning save.
 *
 * =========================================================
 */



import {
    runPlan
} from "../taskRunner.js";


import {
    composeAnswer
} from "../answerComposer.js";


import {
    validateResult
} from "../validator.js";


import {
    analyzeRunFailure
} from "./runFailurePolicy.js";


import {
    buildCompletedResult
} from "./executionResult.js";


import {
    registerExecutionStep,
    registerExecutionFailure
} from "./executionContext.js";









/*
 * =========================================================
 * LEARNING SIGNAL
 * =========================================================
 */


function addLearningSignal(

    context,

    signal

) {


    if (
        !context?.learningContext
    ) {

        return;

    }



    if (
        !Array.isArray(
            context.learningContext.signals
        )
    ) {

        context.learningContext.signals = [];

    }



    context.learningContext.signals.push({

        ...signal,


        timestamp:

            new Date()
                .toISOString()

    });


}









/*
 * =========================================================
 * FAILURE BUILDER
 * =========================================================
 */


function createFailure(

    stage,

    reason,

    failureType,

    extra = {}

) {


    return {


        stage,


        failureType:

            failureType ||
            `${stage}-failure`,



        reason:

            reason ||
            "Ошибка выполнения",



        ...extra

    };

}









/*
 * =========================================================
 * EXCEPTION
 * =========================================================
 */


function handleException(

    context,

    stage,

    error

) {


    const failure =

        createFailure(

            stage,

            error?.message,

            `${stage}-exception`

        );





    registerExecutionStep(

        context,

        {

            stage,

            status:
                "FAILED",

            error:
                failure.reason

        }

    );





    registerExecutionFailure(

        context,

        failure

    );





    addLearningSignal(

        context,

        {

            type:
                "execution_exception",


            stage,


            reason:
                failure.reason

        }

    );





    return {


        success:false,


        failure

    };

}









/*
 * =========================================================
 * EXPERIENCE META
 * =========================================================
 */


function attachExperience(

    result,

    context

) {


    if (
        !result
    ) {

        return result;

    }



    result.executionMeta = {


        ...(result.executionMeta || {}),



        experience:

        {


            used:

                context
                    ?.experience
                    ?.used === true,



            found:

                context
                    ?.experience
                    ?.found === true,



            skills:

                context
                    ?.experience
                    ?.skills || []

        }


    };



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
     * RUNNER
     * =====================================================
     */


    registerExecutionStep(

        context,

        {

            stage:
                "runner",


            status:
                "RUNNING"

        }

    );






    try {


        context.runResult =

            await runPlan(

                context.plan,

                context.task

            );


    } catch(error) {


        return handleException(

            context,

            "runner",

            error

        );

    }







    const runFailure =

        analyzeRunFailure(

            context.runResult

        );






    if (
        runFailure?.failed === true
    ) {


        registerExecutionStep(

            context,

            {

                stage:
                    "runner",

                status:
                    "FAILED",

                failure:
                    runFailure

            }

        );



        registerExecutionFailure(

            context,

            runFailure

        );



        addLearningSignal(

            context,

            {

                type:
                    "runner_failure",


                failure:
                    runFailure

            }

        );



        return {


            success:false,


            failure:
                runFailure

        };


    }







    registerExecutionStep(

        context,

        {

            stage:
                "runner",

            status:
                "COMPLETED"

        }

    );









    /*
     * =====================================================
     * COMPOSER
     * =====================================================
     */


    registerExecutionStep(

        context,

        {

            stage:
                "composer",

            status:
                "RUNNING"

        }

    );






    try {


        context.answerResult =

            await composeAnswer(

                context.task,

                context.plan,

                context.runResult

            );


    } catch(error) {


        return handleException(

            context,

            "composer",

            error

        );

    }








    if (
        !context.answerResult ||
        context.answerResult.success !== true
    ) {


        const failure =

            createFailure(

                "composer",

                context.answerResult?.text ||

                "Ответ не создан",

                "composer-failure"

            );



        registerExecutionFailure(

            context,

            failure

        );



        registerExecutionStep(

            context,

            {

                stage:
                    "composer",

                status:
                    "FAILED"

            }

        );



        return {


            success:false,


            failure

        };


    }






    registerExecutionStep(

        context,

        {

            stage:
                "composer",

            status:
                "COMPLETED"

        }

    );









    /*
     * =====================================================
     * VALIDATOR
     * =====================================================
     */


    registerExecutionStep(

        context,

        {

            stage:
                "validator",

            status:
                "RUNNING"

        }

    );






    let validation;



    try {


        validation =

            await validateResult(

                context.task,

                context.plan,

                context.runResult,

                context.answerResult

            );


    } catch(error) {


        return handleException(

            context,

            "validator",

            error

        );


    }








    context.validationResult =
        validation;








    if (
        validation?.valid === true
    ) {


        registerExecutionStep(

            context,

            {

                stage:
                    "validator",

                status:
                    "COMPLETED"

            }

        );



        context.learningContext.successful =
            true;




        addLearningSignal(

            context,

            {

                type:
                    "validated_success"

            }

        );





        return {


            success:true,


            result:

                attachExperience(

                    buildCompletedResult(

                        context,

                        context.answerResult,

                        true

                    ),

                    context

                )


        };


    }









    const failure =

        createFailure(

            "validator",

            validation?.reason ||

            "Результат не прошёл проверку",

            "validation-error",

            {

                validation

            }

        );





    registerExecutionFailure(

        context,

        failure

    );





    registerExecutionStep(

        context,

        {

            stage:
                "validator",

            status:
                "FAILED",

            validation

        }

    );





    addLearningSignal(

        context,

        {

            type:
                "validation_failure",


            reason:
                failure.reason

        }

    );






    return {


        success:false,


        failure

    };


}
