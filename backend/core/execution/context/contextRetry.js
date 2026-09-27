/*
 * =========================================================
 * JESSICA CONTEXT RETRY v1
 * =========================================================
 *
 * Управление Retry counters.
 *
 *
 * Отвечает:
 *
 * - увеличение retryCount;
 * - история retry.
 *
 *
 * НЕ:
 *
 * - принимает решение Retry;
 * - запускает выполнение.
 *
 * =========================================================
 */







/*
 * =========================================================
 * REGISTER RETRY
 * =========================================================
 */


export function registerExecutionRetry(

    context

) {


    if (

        !context

    ) {

        return;

    }









    context.retryCount++;









    if (

        !Array.isArray(

            context.retryHistory

        )

    ) {


        context.retryHistory = [];

    }









    context.retryHistory.push({

        retryCount:

            context.retryCount,



        attempt:

            context.attempt,



        timestamp:

            new Date()

                .toISOString()

    });


}
