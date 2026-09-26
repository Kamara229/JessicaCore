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
 * Центральный исполнитель Execution Plan.
 *
 *
 * Flow:
 *
 * Plan
 *   ↓
 * Execution Context
 *   ↓
 * Step Runner
 *   ↓
 * Result
 *
 *
 * Experience:
 *
 * Plan.experience
 *        ↓
 * Execution Context
 *        ↓
 * Execution Trace
 *        ↓
 * Learning
 *
 *
 * НЕ:
 *
 * - выполняет конкретные инструменты;
 * - создаёт ответы;
 * - сохраняет Experience;
 * - создаёт Skills.
 *
 * =========================================================
 */







/*
 * =========================================================
 * BUILD EXPERIENCE META
 * =========================================================
 */


function buildExperienceMeta(
    plan,
    planningContext
) {


    return {


        used:

            plan?.experienceUsed === true,



        skills:

            Array.isArray(
                plan?.experience?.skills
            )

                ? plan.experience.skills

                : [],



        context:

            planningContext?.experience || null


    };

}









/*
 * =========================================================
 * CREATE CONTEXT
 * =========================================================
 */


function createJessicaExecutionContext({

    task,

    plan,

    planningContext

}) {


    return createExecutionContext({

        task,


        plan,



        planningContext,



        experience:

            buildExperienceMeta(

                plan,

                planningContext

            )

    });


}









/*
 * =========================================================
 * EXECUTE PLAN
 * =========================================================
 */


export async function executePlanCycle(

    taskText,

    initialPlan,

    planningContext = {}

) {



    /*
     * =====================================================
     * CREATE EXECUTION CONTEXT
     * =====================================================
     */


    const context =

        createJessicaExecutionContext({

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

            "Jessica execution:",

            {

                attempt,


                max:
                    MAX_EXECUTION_ATTEMPTS,


                experienceUsed:
                    context.experience?.used || false


            }

        );






        /*
         * =================================================
         * RUN STEP
         * =================================================
         */


        const stepResult =

            await executeExecutionStep(

                context

            );







        /*
         * =================================================
         * SUCCESS
         * =================================================
         */


        if (

            stepResult?.success === true

        ) {


            return {


                ...stepResult.result,



                executionMeta: {


                    ...(stepResult.result?.executionMeta || {}),



                    experienceUsed:

                        context.experience?.used === true,



                    experienceSkills:

                        context.experience?.skills || []

                }


            };


        }







        /*
         * =================================================
         * FAILURE
         * =================================================
         */


        lastFailure =

            stepResult?.failure || {


                stage:
                    "execution",


                reason:
                    "Unknown execution failure"

            };








        /*
         * =================================================
         * LIMIT
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
         * FAILURE HANDLER
         * =================================================
         */


        const failureResult =

            await handleExecutionFailure(

                context,

                lastFailure

            );







        if (

            failureResult?.finished === true

        ) {


            return failureResult.result;


        }



        /*
         *
         * продолжаем с новым планом
         *
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
                "Лимит выполнения исчерпан"


        }

    );


}
