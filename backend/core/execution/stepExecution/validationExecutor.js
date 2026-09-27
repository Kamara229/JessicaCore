/*
 * =========================================================
 * JESSICA EXECUTION
 * VALIDATION EXECUTOR v2
 * =========================================================
 *
 * Проверка результата выполнения.
 *
 *
 * Flow:
 *
 * Task
 * Plan
 * Run Result
 * Answer Result
 *        ↓
 * Validator
 *        ↓
 * Validation Result
 *
 *
 * Ответственность:
 *
 * - вызвать Validator;
 * - сохранить Validation Result;
 * - нормализовать ошибку проверки.
 *
 *
 * НЕ:
 *
 * - выполняет Tools;
 * - создаёт Answer;
 * - делает Retry;
 * - делает Replan.
 *
 * =========================================================
 */


import {
    validateResult
} from "../../validator.js";









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
 * NORMALIZE VALIDATION FAILURE
 * =========================================================
 */


function normalizeValidationFailure(

    validation

) {


    return {


        stage:

            validation?.stage ||

            "validator",



        failureType:

            validation?.failureType ||

            "validation-error",



        reason:

            validation?.reason ||

            "Результат не прошёл проверку",



        shouldRetry:

            validation?.shouldRetry === true,



        needsClarification:

            validation?.needsClarification === true,



        noVerifiedResult:

            validation?.noVerifiedResult === true,



        validation

    };

}









/*
 * =========================================================
 * EXECUTE VALIDATION
 * =========================================================
 */


export async function executeValidation(

    context

) {


    if (

        !context

    ) {


        return buildFailure(

            "validator",

            "missing-context",

            "Execution context отсутствует"

        );

    }









    let validation;



    try {


        validation =

            await validateResult(

                context.task,

                context.plan,

                context.runResult,

                context.answerResult

            );


    }

    catch(error){


        return buildFailure(

            "validator",

            "validator-exception",

            error?.message ||

            "Ошибка проверки результата"

        );


    }









    context.validationResult =

        validation;









    if (

        validation?.valid !== true

    ) {


        return {


            success:false,


            failure:

                normalizeValidationFailure(

                    validation

                )


        };


    }









    return {


        success:true,


        validationResult:

            validation


    };


}
