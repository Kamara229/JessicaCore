/*
 * =========================================================
 * JESSICA CONTEXT FAILURES v1
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
 * REGISTER FAILURE
 * =========================================================
 */


export function registerExecutionFailure(

    context,

    failure

) {


    if (

        !context ||

        !failure

    ) {

        return;

    }









    /*
     * Последняя ошибка
     */


    context.lastFailure =

        failure;









    /*
     * История ошибок
     */


    if (

        !Array.isArray(

            context.errors

        )

    ) {


        context.errors = [];

    }









    context.errors.push({

        ...failure,



        timestamp:

            new Date()

                .toISOString()


    });


}
