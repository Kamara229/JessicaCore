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
 * Центральный координатор создания плана.
 *
 *
 * Flow:
 *
 * Task
 *   ↓
 * Planning Context
 *   ↓
 * Experience Context
 *   ↓
 * Planner Request
 *   ↓
 * Parse
 *   ↓
 * Normalize
 *   ↓
 * Validate
 *   ↓
 * Execution
 *
 *
 * НЕ:
 *
 * - ищет Experience;
 * - сохраняет Skill;
 * - выполняет задачи;
 * - работает с Tools.
 *
 * =========================================================
 */





/*
 * =========================================================
 * NORMALIZE EXPERIENCE CONTEXT
 * =========================================================
 */


function enrichExperienceContext(
    context
) {


    const experience =
        context?.experience;



    if (
        !experience ||
        typeof experience !== "object"
    ) {


        return {


            experience:


                {


                    available:
                        false,


                    skills:
                        []


                }


        };

    }





    const skills =

        Array.isArray(
            experience?.experience?.skills
        )

            ? experience.experience.skills

            : [];





    return {


        experience:


            {


                available:
                    experience.found === true,


                source:
                    experience.source || "unknown",


                confidence:
                    Number(
                        experience.confidence || 0
                    ),


                skills


            }


    };

}









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



    const experienceContext =
        enrichExperienceContext(
            normalized
        );



    return {


        ...normalized,


        ...experienceContext,


        metadata:


            {


                ...(normalized.metadata || {}),


                experienceAvailable:
                    experienceContext
                        .experience
                        .available,


                experienceSkillCount:
                    experienceContext
                        .experience
                        .skills
                        .length


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







    let planningContext =
        buildPlannerContext(
            context
        );





    let lastError =
        "";








    for (

        let attempt = 1;

        attempt <= MAX_PLANNER_ATTEMPTS;

        attempt++

    ) {




        planningContext = {


            ...planningContext,


            metadata:


                {


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





            if (
                !plan
            ) {


                throw new Error(
                    "normalize_failed"
                );

            }








            /*
             * =================================================
             * EXPERIENCE MARKING
             * =================================================
             *
             * Помечаем:
             *
             * использовался ли опыт.
             *
             * =================================================
             */


            plan.experience = {


                used:

                    planningContext
                        ?.experience
                        ?.available === true,



                source:

                    planningContext
                        ?.experience
                        ?.source ||
                    null,



                skills:

                    planningContext
                        ?.experience
                        ?.skills
                        ?.map(

                            skill =>
                                skill.id ||
                                skill.name

                        )
                        ||
                        []

            };








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


                    steps:
                        plan.steps.length,


                    experienceUsed:
                        plan.experience.used


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








    return {


        success:
            false,


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
