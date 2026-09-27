/*
 * =========================================================
 * JESSICA RESULT BUILDERS v2
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
 * SAFE VALUE
 * =========================================================
 */


function safeString(

    value

) {


    if (

        value === null ||

        value === undefined

    ) {

        return "";

    }


    return String(

        value

    )
    .trim();

}









/*
 * =========================================================
 * NORMALIZE FAILURE
 * =========================================================
 */


function buildFailureMeta(

    {

        stage = "execution",

        reason = "Ошибка выполнения",

        failureType = "execution-error"

    } = {}

) {


    return {


        stage,


        type:

            failureType,


        reason:

            safeString(

                reason

            )


    };

}









/*
 * =========================================================
 * COMPLETED RESULT
 * =========================================================
 */


export function buildCompletedResult(

    context,

    answerResult,

    verified = true

) {


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



        failure:

            null,



        clarification:

            null


    };


}









/*
 * =========================================================
 * FAILED RESULT
 * =========================================================
 */


export function buildFailureResult(

    context,

    options = {}

) {


    const failure =

        buildFailureMeta(

            options

        );





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



        clarification:

            null,



        failure


    };


}









/*
 * =========================================================
 * CLARIFICATION RESULT
 * =========================================================
 */


export function buildClarificationResult(

    context,

    {

        stage = "execution",

        reason = "Требуется уточнение"

    } = {}

) {


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



        failure:

            null,



        clarification:

        {


            stage,


            reason:

                safeString(

                    reason

                )


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

    {

        stage = "verification",

        reason = "Не удалось подтвердить результат",

        failureType = "no-verified-result"

    } = {}

) {


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



        clarification:

            null,



        failure:

            buildFailureMeta({

                stage,

                reason,

                failureType

            })


    };


}
