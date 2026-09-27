/*
 * =========================================================
 * JESSICA CONTEXT ATTEMPTS v3
 * =========================================================
 *
 * Управление попытками Execution.
 *
 *
 * Отвечает:
 *
 * - увеличение счётчика Execution Attempt;
 * - регистрация истории попыток.
 *
 *
 * НЕ:
 *
 * - принимает решение Retry;
 * - регистрирует Retry;
 * - принимает решение Replan;
 * - регистрирует Replan;
 * - создаёт планы;
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
 * REGISTER ATTEMPT
 * =========================================================
 */


export function registerExecutionAttempt(

    context,

    data = {}

) {


    if(

        !context

    ){


        return null;

    }









    /*
     * =====================================================
     * INCREMENT
     * =====================================================
     */


    context.attempt =

        safeNumber(

            context.attempt

        )

        +

        1;









    /*
     * =====================================================
     * HISTORY
     * =====================================================
     */


    if(

        !Array.isArray(

            context.attempts

        )

    ){


        context.attempts = [];

    }









    /*
     * =====================================================
     * RECORD
     * =====================================================
     *
     * Системные поля располагаются после ...data,
     * чтобы внешний caller не мог подменить:
     *
     * - attempt;
     * - retryCount;
     * - replanCount;
     * - timestamp.
     *
     * =====================================================
     */


    const record = {


        ...data,



        attempt:

            context.attempt,



        retryCount:

            safeNumber(

                context.retryCount

            ),



        replanCount:

            safeNumber(

                context.replanCount

            ),



        timestamp:

            new Date()

                .toISOString()


    };









    context.attempts.push(

        record

    );









    return record;

}
