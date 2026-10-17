/*
 * =========================================================
 * JESSICA EXECUTION
 * VALIDATION EXECUTOR v3
 * =========================================================
 *
 * Проверка результата выполнения.
 *
 * Дополнительно сохраняет failureScope/category,
 * чтобы presentation failure не уходил в Execution Replan.
 *
 * =========================================================
 */


import {
    validateResult
} from "../../validator.js";


function buildFailure(

    stage,

    failureType,

    reason,

    extra = {}

) {

    return {

        success:
            false,

        failure: {

            stage,

            failureType,

            reason,

            ...extra

        }

    };

}


function getFailureCategory(
    validation
) {

    const scope =
        String(
            validation?.failureScope || ""
        )
            .trim()
            .toLowerCase();


    if(
        scope === "presentation"
    ){

        return "presentation";

    }


    if(
        scope === "clarification"
    ){

        return "clarification";

    }


    return "validation";

}


function getFailureType(
    validation
) {

    const scope =
        String(
            validation?.failureScope || ""
        )
            .trim()
            .toLowerCase();


    if(
        scope === "presentation"
    ){

        return "answer-validation-error";

    }


    if(
        scope === "evidence"
    ){

        return "evidence-validation-error";

    }


    if(
        scope === "clarification"
    ){

        return "clarification-required";

    }


    return "validation-error";

}


function normalizeValidationFailure(
    validation
) {

    const category =
        getFailureCategory(
            validation
        );


    return {

        stage:
            validation?.stage
            ||
            "validator",


        failureType:
            validation?.failureType
            ||
            getFailureType(
                validation
            ),


        category,


        failureScope:
            validation?.failureScope
            ||
            null,


        reason:
            validation?.reason
            ||
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


export async function executeValidation(
    context
) {

    if(
        !context
    ){

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

    } catch(error){

        return buildFailure(

            "validator",

            "validator-exception",

            error?.message
            ||
            "Ошибка проверки результата"

        );

    }


    context.validationResult =
        validation;


    if(
        validation?.valid !== true
    ){

        return {

            success:
                false,

            failure:
                normalizeValidationFailure(
                    validation
                )

        };

    }


    return {

        success:
            true,

        validationResult:
            validation

    };

}
