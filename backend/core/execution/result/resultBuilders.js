/*
 * =========================================================
 * JESSICA RESULT BUILDERS v3
 * =========================================================
 *
 * Создание Execution Result объектов.
 *
 *
 * Ответственность:
 *
 * - COMPLETED;
 * - FAILED;
 * - NEEDS_CLARIFICATION;
 * - NO_VERIFIED_RESULT.
 *
 *
 * НЕ:
 *
 * - анализирует ошибки;
 * - принимает решения;
 * - выполняет Execution;
 * - изменяет Context.
 *
 * =========================================================
 */


import {
    EXECUTION_RESULT_STATUS
} from "./resultStatus.js";


import {
    buildBaseResult
} from "./resultMeta.js";









/*
 * =========================================================
 * SAFE STRING
 * =========================================================
 */


function safeString(

    value

){

    return String(

        value || ""

    )
    .trim();

}









/*
 * =========================================================
 * FAILURE META
 * =========================================================
 */


function buildFailureMeta(

    options = {}

){

    return {


        stage:

            options.stage ||

            "execution",



        failureType:

            options.failureType ||

            options.type ||

            "execution-error",



        category:

            options.category ||

            "execution",



        reason:

            safeString(

                options.reason

            )
            ||

            "Ошибка выполнения",



        validation:

            options.validation ||

            null,



        source:

            options.source ||

            null,



        details:

            options.details ||

            null



    };

}









/*
 * =========================================================
 * COMPLETED
 * =========================================================
 */


export function buildCompletedResult(

    context,

    answerResult,

    verified = true

){


    return {


        ...buildBaseResult(

            context

        ),



        success:

            true,



        status:

            EXECUTION_RESULT_STATUS.COMPLETED,



        verified:

            verified === true,



        answer:

        {

            text:

                safeString(

                    answerResult?.text

                ),



            source:

                answerResult?.source ||

                "unknown"


        },



        validation:

            context?.validationResult ||

            null,



        failure:

            null,



        clarification:

            null,



        terminal:

            null


    };


}









/*
 * =========================================================
 * FAILED
 * =========================================================
 */


export function buildFailureResult(

    context,

    options = {}

){


    return {


        ...buildBaseResult(

            context

        ),



        success:

            false,



        status:

            EXECUTION_RESULT_STATUS.FAILED,



        verified:

            false,



        answer:

            null,



        validation:

            context?.validationResult ||

            null,



        clarification:

            null,



        terminal:

            {

                type:

                    "FAILURE"

            },



        failure:

            buildFailureMeta(

                options

            )


    };


}









/*
 * =========================================================
 * NEEDS CLARIFICATION
 * =========================================================
 */


export function buildClarificationResult(

    context,

    options = {}

){


    const failure =

        buildFailureMeta(

            {

                ...options,


                category:

                    "clarification"


            }

        );









    return {


        ...buildBaseResult(

            context

        ),



        success:

            false,



        status:

            EXECUTION_RESULT_STATUS
                .NEEDS_CLARIFICATION,



        verified:

            false,



        answer:

            null,



        validation:

            null,



        failure:

            null,



        clarification:

        {


            stage:

                failure.stage,



            reason:

                failure.reason


        },



        terminal:

        {

            type:

                "CLARIFICATION"

        }


    };


}









/*
 * =========================================================
 * NO VERIFIED RESULT
 * =========================================================
 */


export function buildNoVerifiedResult(

    context,

    options = {}

){


    const failure =

        buildFailureMeta(

            {

                ...options,


                category:

                    "no_verified"


            }

        );









    return {


        ...buildBaseResult(

            context

        ),



        success:

            false,



        status:

            EXECUTION_RESULT_STATUS
                .NO_VERIFIED_RESULT,



        verified:

            false,



        answer:

            null,



        validation:

            context?.validationResult ||

            null,



        clarification:

            null,



        terminal:

        {

            type:

                "NO_VERIFIED_RESULT"

        },



        failure


    };


}
