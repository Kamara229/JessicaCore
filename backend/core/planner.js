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
 * JESSICA PLANNER
 * =========================================================
 *
 * Центральный координатор Planner.
 *
 *
 * Flow:
 *
 * Task
 *  ↓
 * PlanningContext
 *  ↓
 * Planner Request
 *  ↓
 * Parse
 *  ↓
 * Normalize
 *  ↓
 * Validate
 *  ↓
 * Retry
 *
 *
 * Не содержит:
 *
 * - AI prompt;
 * - tools;
 * - Experience search;
 * - execution;
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

            success:
                false,

            text:
                "Задача Planner пустая"

        };

    }



    if (
        !process.env.GROQ_API_KEY
    ) {

        return {

            success:
                false,

            text:
                "GROQ_API_KEY отсутствует"

        };

    }



    /*
     * Единый формат контекста.
     */


    let planningContext =
        normalizePlanningContext(
            context
        );



    let lastError =
        "";



    for (
        let attempt = 1;

        attempt <= MAX_PLANNER_ATTEMPTS;

        attempt++

    ) {



        /*
         * Добавляем служебную информацию
         * о попытке.
         */


        planningContext = {

            ...planningContext,


            metadata: {

                ...(planningContext.metadata || {}),

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


            const rawText =
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


            const rawPlan =
                parsePlan(
                    rawText
                );



            /*
             * =================================================
             * NORMALIZE
             * =================================================
             */


            const plan =
                normalizePlan(
                    rawPlan
                );



            if (!plan) {

                throw new Error(
                    "normalize_failed"
                );

            }



            /*
             * =================================================
             * VALIDATE
             * =================================================
             */


            const validation =
                validatePlan(
                    plan
                );



            if (
                !validation.success
            ) {


                lastError =
                    validation.text ||
                    "validation_failed";



                console.warn(

                    "Jessica Planner rejected:",

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
                        plan.steps.length


                }

            );



            return {

                success:
                    true,

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

                attempt <
                MAX_PLANNER_ATTEMPTS

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



    return {

        success:
            false,

        text:
            (
                "Planner не смог создать корректный план: "
                +
                lastError
            )

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
