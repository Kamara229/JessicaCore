/*
 * =========================================================
 * JESSICA CONTEXT FAILURES v2
 * =========================================================
 *
 * Управление ошибками Execution Context.
 *
 *
 * Отвечает:
 *
 * - сохранение последней ошибки;
 * - накопление истории ошибок.
 *
 *
 * НЕ:
 *
 * - анализирует ошибки;
 * - принимает решение Retry/Replan;
 * - меняет Execution Flow.
 *
 * =========================================================
 */







/*
 * =========================================================
 * NORMALIZE FAILURE
 * =========================================================
 */


function normalizeFailure(

    failure

) {


    if(

        !failure ||

        typeof failure !== "object"

    ){

        return {


            stage:

                "execution",



            failureType:

                "unknown",



            reason:

                String(

                    failure ||

                    "Неизвестная ошибка"

                ),



            category:

                "execution",



            validation:

                null,



            needsClarification:

                false,



            noVerifiedResult:

                false


        };

    }









    return {


        stage:

            failure.stage ||

            "execution",



        failureType:

            failure.failureType ||

            "execution-error",



        reason:

            failure.reason ||

            "Ошибка выполнения",



        category:

            failure.category ||

            "execution",



        validation:

            failure.validation ||

            null,



        needsClarification:

            failure.needsClarification === true,



        noVerifiedResult:

            failure.noVerifiedResult === true


    };

}









/*
 * =========================================================
 * REGISTER FAILURE
 * =========================================================
 */


export function registerExecutionFailure(

    context,

    failure

) {


    if(

        !context

    ){

        return null;

    }









    const record = {


        ...normalizeFailure(

            failure

        ),



        timestamp:

            new Date()

                .toISOString()


    };









    /*
     * Last failure
     */


    context.lastFailure =

        record;









    /*
     * History
     */


    if(

        !Array.isArray(

            context.errors

        )

    ){

        context.errors = [];

    }









    context.errors.push(

        record

    );









    /*
     * Counter
     */


    context.failureCount =

        Number(

            context.failureCount || 0

        )

        +

        1;









    return record;

}
