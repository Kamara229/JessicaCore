/*
 * =========================================================
 * JESSICA CONTEXT RETRY v2
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
 * - проверяет лимиты;
 * - запускает выполнение.
 *
 * =========================================================
 */







/*
 * =========================================================
 * SAFE NUMBER
 * =========================================================
 */


function safeNumber(

    value

) {


    const number =

        Number(value);



    return Number.isFinite(number)

        ?

        number

        :

        0;

}









/*
 * =========================================================
 * REGISTER RETRY
 * =========================================================
 */


export function registerExecutionRetry(

    context,

    failure = null

) {


    if(

        !context

    ){

        return null;

    }









    context.retryCount =

        safeNumber(

            context.retryCount

        )

        +

        1;









    if(

        !Array.isArray(

            context.retryHistory

        )

    ){

        context.retryHistory = [];

    }









    const record = {


        retryCount:

            context.retryCount,



        attempt:

            safeNumber(

                context.attempt

            ),



        failure:

            failure || null,



        timestamp:

            new Date()

                .toISOString()


    };









    context.retryHistory.push(

        record

    );









    return record;

}
