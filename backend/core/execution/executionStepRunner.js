/*
 * =========================================================
 * JESSICA EXECUTION STEP RUNNER v6
 * =========================================================
 *
 * Выполняет один Execution Pass.
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
 * Ответственность:
 *
 * - выполнить текущий маршрут;
 * - собрать результат;
 * - вернуть Failure или Success.
 *
 *
 * НЕ:
 *
 * - Retry;
 * - Replan;
 * - Terminal decision;
 * - Learning save;
 * - изменение Execution Flow.
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
    registerExecutionStep
} from "./executionContext.js";









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


function createExceptionFailure(

    stage,

    error

) {


    return createFailure(

        stage,

        error?.message ||

        "Execution exception",

        `${stage}-exception`

    );


}









/*
 * =========================================================
 * STEP REGISTER
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
 * EXPERIENCE META
 * =========================================================
 */


function attachExperience(

    result,

    context

) {


    if(
        !result
    ){

        return result;

    }




    result.executionMeta = {


        ...(result.executionMeta || {}),



        experience:

        {


            used:

                context?.experience?.used === true,



            found:

                context?.experience?.found === true,



            skills:

                context?.experience?.skills || []

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



    markStep(

        context,

        "runner",

        "RUNNING"

    );





    try {


        context.runResult =

            await runPlan(

                context.plan,

                context.task

            );


    } catch(error) {


        const failure =

            createExceptionFailure(

                "runner",

                error

            );



        markStep(

            context,

            "runner",

            "FAILED",

            {

                failure

            }

        );



        return {


            success:false,


            failure

        };


    }









    const analyzedFailure =

        analyzeRunFailure(

            context.runResult

        );









    if(

        analyzedFailure?.failed === true

    ){


        markStep(

            context,

            "runner",

            "FAILED",

            {

                failure:

                    analyzedFailure

            }

        );



        return {


            success:false,


            failure:

                analyzedFailure


        };


    }









    markStep(

        context,

        "runner",

        "COMPLETED"

    );









    /*
     * =====================================================
     * COMPOSER
     * =====================================================
     */



    markStep(

        context,

        "composer",

        "RUNNING"

    );






    try {


        context.answerResult =

            await composeAnswer(

                context.task,

                context.plan,

                context.runResult

            );



    } catch(error) {



        const failure =

            createExceptionFailure(

                "composer",

                error

            );



        markStep(

            context,

            "composer",

            "FAILED",

            {

                failure

            }

        );



        return {


            success:false,


            failure

        };


    }









    if(

        !context.answerResult ||

        context.answerResult.success !== true

    ){


        const failure =

            createFailure(

                "composer",

                context.answerResult?.text ||

                "Ответ не создан",

                "composer-failure"

            );



        markStep(

            context,

            "composer",

            "FAILED",

            {

                failure

            }

        );



        return {


            success:false,


            failure

        };


    }









    markStep(

        context,

        "composer",

        "COMPLETED"

    );









    /*
     * =====================================================
     * VALIDATOR
     * =====================================================
     */



    markStep(

        context,

        "validator",

        "RUNNING"

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


        const failure =

            createExceptionFailure(

                "validator",

                error

            );



        markStep(

            context,

            "validator",

            "FAILED",

            {

                failure

            }

        );



        return {


            success:false,


            failure

        };


    }









    context.validationResult =

        validation;









    if(

        validation?.valid === true

    ){


        markStep(

            context,

            "validator",

            "COMPLETED"

        );





        const result =

            buildCompletedResult(

                context,

                context.answerResult,

                true

            );





        return {


            success:true,


            result:

                attachExperience(

                    result,

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









    markStep(

        context,

        "validator",

        "FAILED",

        {

            validation,

            failure

        }

    );






    return {


        success:false,


        failure

    };


}
