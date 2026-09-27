/*
 * =========================================================
 * JESSICA CONTEXT ATTEMPTS v1
 * =========================================================
 *
 * Управление попытками Execution.
 *
 *
 * Отвечает:
 *
 * - регистрация попыток;
 * - счётчики retry;
 * - счётчики replan;
 * - история перепланирования.
 *
 *
 * НЕ:
 *
 * - принимает решение Retry/Replan;
 * - создаёт планы;
 * - запускает выполнение.
 *
 * =========================================================
 */







/*
 * =========================================================
 * REGISTER ATTEMPT
 * =========================================================
 */


export function registerExecutionAttempt(

    context,

    data = {}

) {


    if (

        !context

    ) {

        return;

    }









    context.attempt++;









    if (

        !Array.isArray(

            context.attempts

        )

    ) {


        context.attempts = [];

    }









    context.attempts.push({

        attempt:

            context.attempt,



        ...data,



        timestamp:

            new Date()

                .toISOString()


    });


}









/*
 * =========================================================
 * REGISTER REPLAN
 * =========================================================
 */


export function registerExecutionReplan(

    context,

    data = {}

) {


    if (

        !context

    ) {

        return;

    }









    context.replanCount++;









    const record = {


        ...data,



        timestamp:

            new Date()

                .toISOString()


    };









    if (

        !Array.isArray(

            context.replanHistory

        )

    ) {


        context.replanHistory = [];

    }









    context.replanHistory.push(

        record

    );









    if (

        !Array.isArray(

            context.replans

        )

    ) {


        context.replans = [];

    }









    context.replans.push(

        record

    );


}
