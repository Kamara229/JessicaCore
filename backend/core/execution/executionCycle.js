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
 * Оркестратор выполнения плана.
 *
 *
 * Flow:
 *
 * Plan
 *  ↓
 * Execution Step
 *  ↓
 * Failure Handler
 *  ↓
 * Retry / Replan
 *  ↓
 * Result
 *
 *
 * НЕ содержит:
 *
 * - Runner logic
 * - Composer logic
 * - Validator logic
 * - Retry rules
 * - Replan logic
 *
 * =========================================================
 */





export async function executePlanCycle(

    taskText,

    initialPlan,

    planningContext = {}

) {



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
         * Один полный шаг:
         *
         * Runner
         * Composer
         * Validator
         */


        const step =
            await executeExecutionStep(
                context
            );





        if (
            step.success
        ) {


            return step.result;


        }





        /*
         * Ошибка выполнения
         */


        lastFailure =
            step.failure;





        const failureResult =
            await handleExecutionFailure(

                context,

                step.failure

            );





        if (
            failureResult.finished
        ) {


            return failureResult.result;


        }



    }





    /*
     * Все попытки закончились
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
