/*
 * =========================================================
 * JESSICA EXECUTION
 * VALIDATION EXECUTOR v1
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


        return buildFailure(

            "validator",

            "validation-error",

            validation?.reason ||

            "Результат не прошёл проверку",


            {

                validation

            }

        );


    }









    return {


        success:true,


        validationResult:

            validation


    };


}
