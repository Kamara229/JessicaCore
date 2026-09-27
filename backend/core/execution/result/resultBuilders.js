/*
 * =========================================================
 * JESSICA RESULT BUILDERS v1
 * =========================================================
 *
 * Создание Execution Result объектов.
 *
 *
 * Отвечает:
 *
 * - COMPLETED
 * - FAILED
 * - NEEDS_CLARIFICATION
 * - NO_VERIFIED_RESULT
 *
 *
 * НЕ:
 *
 * - анализирует ошибки;
 * - принимает решения;
 * - выполняет Execution.
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

) {

    return String(
        value || ""
    )
    .trim();

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

    {

        stage = "execution",

        reason = "Не удалось выполнить задачу",

        failureType = "execution_failure"

    } = {}

) {


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



        failure:

        {


            stage,


            type:

                failureType,



            reason:

                safeString(

                    reason

                )


        }


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

        failureType = "no_verified_result"

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

        {


            stage,


            type:

                failureType,



            reason:

                safeString(

                    reason

                )


        }


    };


}
