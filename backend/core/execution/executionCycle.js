import {
    createExecutionContext
} from "./executionContext.js";


import {
    executeExecutionStep
} from "./executionStepRunner.js";


import {
    handleExecutionFailure
} from "./executionFailureHandler.js";


import {
    buildTerminalResult
} from "./executionTerminal.js";


import {
    MAX_EXECUTION_ATTEMPTS
} from "./retryPolicy.js";



/*
 * =========================================================
 * JESSICA EXECUTION CYCLE
 * =========================================================
 *
 * Центральный цикл выполнения плана.
 *
 *
 * Flow:
 *
 * Plan
 *   ↓
 * Execution Step
 *   ↓
 * Success
 *
 * или
 *
 * Failure
 *   ↓
 * Failure Handler
 *   ↓
 * Retry / Replan
 *
 *
 * Этот файл НЕ содержит:
 *
 * - TaskRunner;
 * - Answer Composer;
 * - Validator;
 * - Retry logic;
 * - Replan logic;
 * - Terminal logic.
 *
 * =========================================================
 */





export async function executePlanCycle(

    taskText,

    initialPlan,

    planningContext = {}

) {



    /*
     * =====================================================
     * CONTEXT
     * =====================================================
     */


    const context =
        createExecutionContext({

            task:
                taskText,

            plan:
                initialPlan,

            planningContext

        });





    let lastFailure =
        null;






    /*
     * =====================================================
     * EXECUTION LOOP
     * =====================================================
     */


    for (

        let attempt = 1;

        attempt <= MAX_EXECUTION_ATTEMPTS;

        attempt++

    ) {



        context.attempt =
            attempt;




        console.log(

            `Jessica execution attempt ${attempt}/${MAX_EXECUTION_ATTEMPTS}`

        );






        /*
         * =================================================
         * STEP
         * =================================================
         */


        const step =
            await executeExecutionStep(

                context

            );






        /*
         * =================================================
         * SUCCESS
         * =================================================
         */


        if (
            step.success === true
        ) {


            return step.result;


        }






        /*
         * =================================================
         * FAILURE
         * =================================================
         */


        lastFailure =
            step.failure;






        /*
         * =================================================
         * ATTEMPTS LIMIT
         * =================================================
         */


        if (

            attempt >= MAX_EXECUTION_ATTEMPTS

        ) {


            return buildTerminalResult(

                context,

                lastFailure

            );


        }






        /*
         * =================================================
         * HANDLE FAILURE
         * =================================================
         */


        const failureResult =
            await handleExecutionFailure(

                context,

                lastFailure

            );






        /*
         * =================================================
         * FINAL RESULT
         * =================================================
         */


        if (

            failureResult.finished === true

        ) {


            return failureResult.result;


        }




        /*
         * иначе продолжаем цикл
         *
         * с новым планом после replan
         */

    }







    /*
     * =====================================================
     * FALLBACK
     * =====================================================
     */


    return buildTerminalResult(

        context,

        lastFailure || {

            stage:
                "execution",


            failureType:
                "execution-limit",


            reason:
                "Исчерпан лимит выполнения"

        }

    );


}
