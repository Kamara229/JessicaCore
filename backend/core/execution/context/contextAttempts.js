/*
 * =========================================================
 * JESSICA CONTEXT ATTEMPTS v2
 * =========================================================
 *
 * Управление попытками Execution.
 *
 *
 * Отвечает:
 *
 * - регистрация Execution Attempt;
 * - история попыток.
 *
 *
 * НЕ:
 *
 * - принимает решение Retry;
 * - принимает решение Replan;
 * - создаёт планы;
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









    context.attempt =

        safeNumber(

            context.attempt

        )

        +

        1;









    if(

        !Array.isArray(

            context.attempts

        )

    ){

        context.attempts = [];

    }









    const record = {


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



        ...data,



        timestamp:

            new Date()

                .toISOString()


    };









    context.attempts.push(

        record

    );









    return record;

}
