/*
 * =========================================================
 * JESSICA CONTEXT RETRY v3
 * =========================================================
 *
 * Управление Retry-состоянием Execution Context.
 *
 *
 * Отвечает:
 *
 * - увеличение retryCount;
 * - регистрация retryHistory;
 * - связь Retry с Failure и текущим Attempt.
 *
 *
 * НЕ:
 *
 * - принимает решение Retry;
 * - проверяет Retry limits;
 * - классифицирует Failure;
 * - запускает Execution.
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

        Number(

            value

        );



    return Number.isFinite(

        number

    )

        ?

        number

        :

        0;

}









/*
 * =========================================================
 * ENSURE HISTORY
 * =========================================================
 */


function ensureRetryHistory(

    context

) {


    if(

        !Array.isArray(

            context.retryHistory

        )

    ){


        context.retryHistory = [];

    }



    return context.retryHistory;

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









    /*
     * =====================================================
     * COUNTER
     * =====================================================
     */


    context.retryCount =

        safeNumber(

            context.retryCount

        )

        +

        1;









    /*
     * =====================================================
     * RECORD
     * =====================================================
     */


    const record = {


        retryCount:

            context.retryCount,



        attempt:

            safeNumber(

                context.attempt

            ),



        replanCount:

            safeNumber(

                context.replanCount

            ),



        failure:

            failure ||

            null,



        timestamp:

            new Date()

                .toISOString()


    };









    /*
     * =====================================================
     * HISTORY
     * =====================================================
     */


    const history =

        ensureRetryHistory(

            context

        );



    history.push(

        record

    );









    return record;

}
