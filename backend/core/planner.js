/*
 * =========================================================
 * JESSICA PLANNER CORE v3
 * =========================================================
 *
 * Центральный координатор создания Execution Plan.
 *
 *
 * Flow:
 *
 * Task
 *   ↓
 * Planning Context
 *   ↓
 * Planner Request
 *   ↓
 * AI Response
 *   ↓
 * Parse
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
 * BUILD CONTEXT
 * =========================================================
 */


function buildPlannerContext(
    context
) {


    const normalized =

        normalizePlanningContext(
            context
        );



    return {


        ...normalized,


        metadata:


        {


            ...(normalized.metadata || {}),


            planner:

            {


                version:
                    "3",


                createdAt:
                    new Date()
                        .toISOString()


            }


        }


    };


}









/*
 * =========================================================
 * ADD PLAN META
 * =========================================================
 */


function enrichPlan(
    plan,
    context,
    attempt
) {


    return {


        ...plan,


        metadata:


        {


            ...(plan.metadata || {}),


            plannerAttempt:
                attempt,


            generatedAt:
                new Date()
                    .toISOString()


        },



        experience:


            {


                ...(plan.experience || {}),



                available:

                    Boolean(
                        context?.experience
                    )

            }


    };


}









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






    if (
        !cleanTask
    ) {


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







    const baseContext =

        buildPlannerContext(
            context
        );






    let lastError =
        "";





    const trace = [];


    





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
             * REQUEST
             */


            const rawResponse =

                await requestPlan(

                    cleanTask,

                    lastError,

                    planningContext

                );






            trace.push({

                stage:
                    "request",


                attempt

            });










            /*
             * PARSE
             */


            const parsedPlan =

                parsePlan(
                    rawResponse
                );



            if (
                !parsedPlan
            ) {


                throw new Error(
                    "planner_parse_failed"
                );

            }





            trace.push({

                stage:
                    "parse",


                success:
                    true

            });









            /*
             * NORMALIZE
             */


            const normalizedPlan =

                normalizePlan(
                    parsedPlan
                );



            if (
                !normalizedPlan
            ) {


                throw new Error(
                    "planner_normalize_failed"
                );

            }







            trace.push({

                stage:
                    "normalize",


                success:
                    true

            });









            /*
             * VALIDATE
             */


            const validation =

                validatePlan(

                    normalizedPlan,

                    planningContext

                );







            if (
                !validation.success
            ) {


                lastError =

                    validation.text ||

                    "planner_validation_failed";





                trace.push({

                    stage:
                        "validation",


                    success:
                        false,


                    reason:
                        lastError


                });





                continue;

            }









            const finalPlan =

                enrichPlan(

                    normalizedPlan,

                    planningContext,

                    attempt

                );









            trace.push({

                stage:
                    "completed",


                success:
                    true

            });









            console.log(

                "Jessica Planner completed:",

                {

                    intent:
                        finalPlan.intent,


                    steps:
                        finalPlan.steps.length,


                    attempt

                }

            );








            return {


                success:
                    true,


                plan:
                    finalPlan,



                context:
                    planningContext,



                trace


            };








        } catch(error) {



            lastError =

                error?.message ||

                "planner_error";






            trace.push({

                stage:
                    "error",


                attempt,


                error:
                    lastError


            });







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



                continue;

            }



        }


    }








    return {


        success:
            false,


        text:

            "Planner не смог создать корректный план: "
            +

            lastError,



        trace


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
