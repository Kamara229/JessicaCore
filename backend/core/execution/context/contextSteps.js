/*
 * =========================================================
 * JESSICA CONTEXT STEPS v1
 * =========================================================
 *
 * Управление историей Execution Steps.
 *
 *
 * Отвечает:
 *
 * - регистрация выполненных этапов;
 * - хранение истории;
 * - фиксация completed/failed шагов.
 *
 *
 * НЕ:
 *
 * - выполняет шаги;
 * - принимает решения;
 * - управляет Execution Flow.
 *
 * =========================================================
 */







/*
 * =========================================================
 * REGISTER STEP
 * =========================================================
 */


export function registerExecutionStep(

    context,

    step

) {


    if (

        !context ||

        !step

    ) {

        return;

    }







    const record = {


        ...step,



        timestamp:

            new Date()

                .toISOString()


    };









    /*
     * Current step
     */


    context.currentStep =

        step.stage ||

        null;









    /*
     * Full history
     */


    if (

        !Array.isArray(

            context.stepsHistory

        )

    ) {


        context.stepsHistory = [];

    }






    context.stepsHistory.push(

        record

    );









    if (

        !Array.isArray(

            context.executionHistory

        )

    ) {


        context.executionHistory = [];

    }






    context.executionHistory.push(

        record

    );









    /*
     * Completed
     */


    if (

        step.status === "COMPLETED"

    ) {


        if (

            !Array.isArray(

                context.completedSteps

            )

        ) {


            context.completedSteps = [];

        }




        context.completedSteps.push(

            record

        );


    }









    /*
     * Failed
     */


    if (

        step.status === "FAILED"

    ) {


        if (

            !Array.isArray(

                context.failedSteps

            )

        ) {


            context.failedSteps = [];

        }




        context.failedSteps.push(

            record

        );


    }

}
