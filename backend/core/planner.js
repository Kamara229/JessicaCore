import {
    requestPlan
} from "./planner/plannerRequest.js";


import {
    parsePlan
} from "./planner/planParser.js";


import {
    normalizePlan
} from "./planner/planNormalizer.js";


import {
    validatePlan
} from "./planner/planValidator.js";


import {
    normalizePlanningContext
} from "./planner/planningContext.js";


import {
    MAX_PLANNER_ATTEMPTS,
    sleep,
    isRetryablePlannerError,
    getPlannerRetryDelay
} from "./planner/plannerRetry.js";





/*
 * =========================================================
 * JESSICA PLANNER CORE v2
 * =========================================================
 *
 * Центральный координатор Planner.
 *
 *
 * Flow:
 *
 * Task
 *   ↓
 * PlanningContext
 *   ↓
 * Planner Request
 *   ↓
 * AI Planner
 *   ↓
 * Parse JSON
 *   ↓
 * Normalize
 *   ↓
 * Validate
 *   ↓
 * Execution Plan
 *
 *
 * НЕ:
 *
 * - ищет Experience;
 * - хранит Skills;
 * - выполняет Tools;
 * - обучает Jessica;
 * - отвечает пользователю.
 *
 * =========================================================
 */






/*
 * =========================================================
 * CREATE PLAN
 * =========================================================
 */


export async function createPlan(

    task,

    context = {}

) {


    const cleanTask =

        String(
            task || ""
        )
        .trim();




    if (!cleanTask) {


        return {

            success:false,

            text:
                "Задача Planner пустая"

        };

    }





    if (
        !process.env.GROQ_API_KEY
    ) {


        return {

            success:false,

            text:
                "GROQ_API_KEY отсутствует"

        };

    }







    /*
     * =====================================================
     * CONTEXT
     * =====================================================
     */


    const baseContext =

        normalizePlanningContext(
            context
        );





    let lastError =
        "";







    /*
     * =====================================================
     * RETRY LOOP
     * =====================================================
     */


    for (

        let attempt = 1;

        attempt <= MAX_PLANNER_ATTEMPTS;

        attempt++

    ) {




        const planningContext = {


            ...baseContext,


            metadata:
            {

                ...(baseContext.metadata || {}),


                plannerAttempt:
                    attempt

            }


        };








        try {





            /*
             * =================================================
             * REQUEST
             * =================================================
             */


            const rawResponse =

                await requestPlan(

                    cleanTask,

                    lastError,

                    planningContext

                );








            /*
             * =================================================
             * PARSE
             * =================================================
             */


            const parsedPlan =

                parsePlan(

                    rawResponse

                );




            if (!parsedPlan) {


                throw new Error(
                    "planner_parse_failed"
                );

            }








            /*
             * =================================================
             * NORMALIZE
             * =================================================
             */


            const plan =

                normalizePlan(

                    parsedPlan

                );




            if (!plan) {


                throw new Error(
                    "planner_normalize_failed"
                );

            }









            /*
             * =================================================
             * VALIDATE
             * =================================================
             */


            const validation =

                validatePlan(

                    plan,

                    planningContext

                );







            if (
                !validation.success
            ) {



                lastError =

                    validation.text ||

                    "planner_validation_failed";




                console.warn(

                    "Jessica Planner validation failed:",

                    {

                        attempt,

                        reason:
                            lastError

                    }

                );



                continue;


            }









            /*
             * =================================================
             * SUCCESS
             * =================================================
             */


            console.log(

                "Jessica Planner success:",

                {

                    intent:
                        plan.intent,


                    requiresTools:
                        plan.requiresTools,


                    steps:
                        plan.steps.length,


                    attempt

                }

            );








            return {


                success:true,


                plan,


                context:
                    planningContext


            };







        } catch(error) {




            lastError =

                error?.message ||

                "planner_error";






            console.error(

                "Jessica Planner error:",

                {

                    attempt,

                    error:
                        lastError

                }

            );






            if (

                attempt < MAX_PLANNER_ATTEMPTS

                &&

                isRetryablePlannerError(
                    error
                )

            ) {


                await sleep(

                    getPlannerRetryDelay(
                        attempt
                    )

                );


            }



        }



    }








    /*
     * =====================================================
     * FAILED
     * =====================================================
     */


    return {


        success:false,


        text:

            "Planner не смог создать корректный план: "

            +

            lastError


    };

}









/*
 * =========================================================
 * LEGACY EXPORT
 * =========================================================
 */


export async function planTask(

    task,

    context = {}

) {


    const result =

        await createPlan(

            task,

            context

        );




    if (
        result.success
    ) {

        return result.plan;

    }




    throw new Error(

        result.text

    );


}
