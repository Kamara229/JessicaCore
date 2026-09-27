/*
 * =========================================================
 * JESSICA CONTEXT STEPS v2
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
 * STATUS
 * =========================================================
 */


const STEP_STATUS = {


    RUNNING:

        "RUNNING",



    COMPLETED:

        "COMPLETED",



    FAILED:

        "FAILED"


};









/*
 * =========================================================
 * SAFE ARRAY
 * =========================================================
 */


function ensureArray(

    object,

    field

) {


    if(

        !Array.isArray(

            object[field]

        )

    ){

        object[field] = [];

    }


    return object[field];

}









/*
 * =========================================================
 * NORMALIZE STEP
 * =========================================================
 */


function normalizeStep(

    step

) {


    return {


        stage:

            step.stage || null,



        status:

            Object.values(

                STEP_STATUS

            )
            .includes(

                step.status

            )

                ?

                step.status

                :

                STEP_STATUS.RUNNING,



        data:

            step.data || {},



        failure:

            step.failure || null



    };

}









/*
 * =========================================================
 * REGISTER STEP
 * =========================================================
 */


export function registerExecutionStep(

    context,

    step

) {


    if(

        !context ||

        !step

    ){

        return null;

    }









    const record = {


        ...normalizeStep(

            step

        ),



        timestamp:

            new Date()

                .toISOString()


    };









    context.currentStep =

        record.stage;









    const steps =

        ensureArray(

            context,

            "stepsHistory"

        );



    steps.push(

        record

    );









    /*
     * Backward compatibility
     */


    context.executionHistory =

        context.stepsHistory;









    if(

        record.status ===

        STEP_STATUS.COMPLETED

    ){


        registerCompletedStep(

            context,

            record

        );


    }









    if(

        record.status ===

        STEP_STATUS.FAILED

    ){


        registerFailedStep(

            context,

            record

        );


    }









    return record;

}









/*
 * =========================================================
 * COMPLETED
 * =========================================================
 */


export function registerCompletedStep(

    context,

    step

) {


    if(

        !context

    ){

        return null;

    }



    const list =

        ensureArray(

            context,

            "completedSteps"

        );



    list.push(

        step

    );


    return step;

}









/*
 * =========================================================
 * FAILED
 * =========================================================
 */


export function registerFailedStep(

    context,

    step

) {


    if(

        !context

    ){

        return null;

    }



    const list =

        ensureArray(

            context,

            "failedSteps"

        );



    list.push(

        step

    );


    return step;

}
