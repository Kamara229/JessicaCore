/*
 * =========================================================
 * JESSICA CONTEXT STEPS v3
 * =========================================================
 *
 * Управление историей Execution Steps.
 *
 *
 * Отвечает:
 *
 * - регистрация Execution Step;
 * - хранение канонической истории stepsHistory;
 * - фиксация completedSteps;
 * - фиксация failedSteps.
 *
 *
 * НЕ:
 *
 * - выполняет шаги;
 * - принимает решения;
 * - управляет Execution Flow;
 * - хранит альтернативную историю шагов.
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
 * ENSURE ARRAY
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


    const source =

        step &&

        typeof step === "object"

            ?

            step

            :

            {};









    const status =

        Object.values(

            STEP_STATUS

        )
        .includes(

            source.status

        )

            ?

            source.status

            :

            STEP_STATUS.RUNNING;









    return {


        stage:

            source.stage ||

            null,



        status,



        data:

            source.data &&

            typeof source.data === "object"

                ?

                source.data

                :

                {},



        failure:

            source.failure ||

            null


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

        !step ||

        typeof step !== "object"

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









    /*
     * =====================================================
     * CURRENT STEP
     * =====================================================
     */


    context.currentStep =

        record.stage;









    /*
     * =====================================================
     * STEP HISTORY
     * =====================================================
     *
     * stepsHistory является единственным
     * каноническим журналом Execution Steps.
     *
     * =====================================================
     */


    const steps =

        ensureArray(

            context,

            "stepsHistory"

        );



    steps.push(

        record

    );









    /*
     * =====================================================
     * COMPLETED
     * =====================================================
     */


    if(

        record.status ===

        STEP_STATUS.COMPLETED

    ){


        registerCompletedStep(

            context,

            record

        );

    }









    /*
     * =====================================================
     * FAILED
     * =====================================================
     */


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
 * REGISTER COMPLETED STEP
 * =========================================================
 */


export function registerCompletedStep(

    context,

    step

) {


    if(

        !context ||

        !step

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
 * REGISTER FAILED STEP
 * =========================================================
 */


export function registerFailedStep(

    context,

    step

) {


    if(

        !context ||

        !step

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
