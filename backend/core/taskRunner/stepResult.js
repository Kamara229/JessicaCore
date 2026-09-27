/*
 * =========================================================
 * JESSICA TASK RUNNER
 * STEP RESULT v2
 * =========================================================
 *
 * Нормализация результата одного Tool шага.
 *
 *
 * Flow:
 *
 * Tool Result
 *      ↓
 * Normalize
 *      ↓
 * TaskRunner Step Result
 *
 *
 * Сохраняет:
 *
 * - identity шага;
 * - arguments;
 * - data;
 * - raw result;
 * - status;
 * - failure metadata.
 *
 *
 * НЕ:
 *
 * - принимает решения;
 * - делает Retry;
 * - делает Replan.
 *
 * =========================================================
 */







/*
 * =========================================================
 * STRING
 * =========================================================
 */


function normalizeString(

    value

) {


    return typeof value === "string"

        ?

        value.trim()

        :

        "";

}









/*
 * =========================================================
 * ARGUMENTS
 * =========================================================
 */


function normalizeArguments(

    value

) {


    if (

        !value

        ||

        typeof value !== "object"

        ||

        Array.isArray(value)

    ) {


        return {};

    }


    return value;

}









/*
 * =========================================================
 * NORMALIZE STEP RESULT
 * =========================================================
 */


export function normalizeStepResult(

    step,

    result

) {


    return {


        /*
         * =================================================
         * STEP IDENTITY
         * =================================================
         */


        id:

            step?.id ||

            null,



        tool:

            step?.tool ||

            "",



        arguments:

            normalizeArguments(

                step?.arguments

            ),









        /*
         * =================================================
         * STATUS
         * =================================================
         */


        success:

            result?.success === true,



        failed:

            result?.success !== true,









        /*
         * =================================================
         * CONTENT
         * =================================================
         */


        text:

            normalizeString(

                result?.text

            ),



        reason:

            normalizeString(

                result?.reason

            ),



        data:

            result?.data ??

            null,



        raw:

            result ?? null,









        /*
         * =================================================
         * ROUTING FLAGS
         * =================================================
         */


        shouldRetry:

            result?.shouldRetry === true,



        needsClarification:

            result?.needsClarification === true,









        /*
         * =================================================
         * FAILURE META
         * =================================================
         */


        stage:

            normalizeString(

                result?.stage

            ),



        failureType:

            normalizeString(

                result?.failureType

            )



    };

}
