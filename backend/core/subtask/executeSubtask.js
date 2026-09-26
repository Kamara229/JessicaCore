import {
    createPlan
} from "../planner.js";


import {
    executePlanCycle
} from "../execution/executionCycle.js";


import {
    resolveExperience
} from "../../experience/experienceCore.js";





/*
 * =========================================================
 * JESSICA EXECUTE SUBTASK
 * =========================================================
 *
 * Выполнение одной подзадачи.
 *
 *
 * Flow:
 *
 * Subtask
 *      ↓
 * Memory Context
 *      ↓
 * Experience Resolver
 *      ↓
 * Planning Context
 *      ↓
 * Planner
 *      ↓
 * Execution Cycle
 *      ↓
 * Result
 *
 *
 * Ответственность:
 *
 * - подготовить Experience Context;
 * - создать план;
 * - запустить выполнение;
 * - вернуть полный результат.
 *
 *
 * НЕ:
 *
 * - выполняет несколько задач;
 * - создаёт Skill;
 * - сохраняет Experience;
 * - управляет Learning.
 *
 * =========================================================
 */





/*
 * =========================================================
 * FAILED RESULT
 * =========================================================
 */


function buildFailedResult({

    subtaskId,

    taskText,

    stage,

    message,

    experience = null,

    plan = null,

    planningContext = null

}) {


    return {


        id:
            subtaskId,


        text:
            taskText,


        success:
            false,


        status:
            "FAILED",


        stage,


        result:
            message,


        experience,


        plan,


        planningContext


    };

}







/*
 * =========================================================
 * BUILD MEMORY EXPERIENCE
 * =========================================================
 *
 * Опыт, который пришёл из Decomposer
 * через Memory Context.
 *
 * =========================================================
 */


function buildMemoryExperience(
    subtask
) {


    const hints =
        Array.isArray(
            subtask?.memoryHints
        )
            ? subtask.memoryHints
            : [];



    if (
        hints.length === 0
    ) {

        return null;

    }



    return {


        source:
            "memory-context",


        skills:
            hints.map(

                item => ({

                    id:
                        item?.id || null,


                    name:
                        item?.name || "",


                    workflow:
                        item?.workflow || [],


                    constraints:
                        item?.constraints || []

                })

            )


    };

}





/*
 * =========================================================
 * MERGE EXPERIENCE
 * =========================================================
 */


function mergeExperienceContext(

    resolverExperience,

    memoryExperience

) {


    if (
        memoryExperience
    ) {


        return {


            found:
                true,


            confidence:
                0.8,


            source:
                "memory-context",


            experience:
                memoryExperience


        };

    }




    return resolverExperience || {


        found:
            false,


        confidence:
            0,


        source:
            "none",


        experience:
            null


    };


}








/*
 * =========================================================
 * EXECUTE SUBTASK
 * =========================================================
 */


export async function executeSubtask(
    subtask
) {


    const subtaskId =
        subtask?.id ?? null;



    const taskText =
        typeof subtask?.text === "string"

            ? subtask.text.trim()

            : "";





    /*
     * =====================================================
     * INPUT
     * =====================================================
     */


    if (
        !taskText
    ) {


        return buildFailedResult({

            subtaskId,

            taskText,

            stage:
                "input",

            message:
                "Подзадача не содержит текста"

        });


    }







    /*
     * =====================================================
     * 1. EXPERIENCE CONTEXT
     * =====================================================
     */


    let resolverResult =
        null;



    let memoryExperience =
        null;



    try {


        memoryExperience =
            buildMemoryExperience(
                subtask
            );




        /*
         * Если Decomposer уже передал опыт,
         * дополнительный поиск не обязателен.
         *
         * Если опыта нет —
         * ищем в Experience Storage.
         */


        if (
            !memoryExperience
        ) {


            resolverResult =
                await resolveExperience(
                    taskText
                );


        }



    } catch(error) {


        console.error(

            `Jessica Experience error ${subtaskId}:`,

            error

        );


        resolverResult = {


            found:
                false,


            confidence:
                0,


            source:
                "error",


            planningContext:
                {}


        };


    }







    const experienceResult =
        mergeExperienceContext(

            resolverResult,

            memoryExperience

        );








    const initialPlanningContext = {


        experience:


            experienceResult,



        subtask:


            {


                id:
                    subtaskId,


                text:
                    taskText


            }



    };






    console.log(

        `Jessica Subtask ${subtaskId} experience:`,

        {


            found:
                experienceResult.found,


            source:
                experienceResult.source,


            confidence:
                experienceResult.confidence


        }

    );








    /*
     * =====================================================
     * 2. CREATE PLAN
     * =====================================================
     */


    let planResult;



    try {


        planResult =
            await createPlan(

                taskText,

                initialPlanningContext

            );



    } catch(error) {


        console.error(

            `Jessica Planner error ${subtaskId}:`,

            error

        );



        return buildFailedResult({

            subtaskId,

            taskText,

            stage:
                "planner",

            message:
                "Ошибка создания плана",

            experience:
                experienceResult,

            planningContext:
                initialPlanningContext

        });


    }







    if (
        !planResult?.success ||
        !planResult?.plan
    ) {


        return buildFailedResult({

            subtaskId,

            taskText,

            stage:
                "planner",

            message:
                planResult?.text ||
                "План не создан",

            experience:
                experienceResult,

            planningContext:
                initialPlanningContext

        });


    }








    /*
     * =====================================================
     * 3. EFFECTIVE CONTEXT
     * =====================================================
     */


    const effectivePlanningContext =

        planResult.context &&

        typeof planResult.context === "object"

            ? planResult.context

            : initialPlanningContext;








    /*
     * =====================================================
     * 4. EXECUTION
     * =====================================================
     */


    let executionResult;



    try {


        executionResult =
            await executePlanCycle(

                taskText,

                planResult.plan,

                effectivePlanningContext

            );



    } catch(error) {


        console.error(

            `Jessica Execution Cycle error ${subtaskId}:`,

            error

        );



        return buildFailedResult({

            subtaskId,

            taskText,

            stage:
                "execution",

            message:
                "Ошибка цикла выполнения",

            experience:
                experienceResult,

            plan:
                planResult.plan,

            planningContext:
                effectivePlanningContext

        });


    }








    /*
     * =====================================================
     * 5. FINAL RESULT
     * =====================================================
     */


    return {


        id:
            subtaskId,


        text:
            taskText,



        experience:


            {


                found:
                    experienceResult.found,


                source:
                    experienceResult.source,


                confidence:
                    experienceResult.confidence,


                skills:

                    experienceResult
                        ?.experience
                        ?.skills || []



            },



        executionMeta:


            {


                experienceUsed:
                    experienceResult.found,


                experienceSource:
                    experienceResult.source,


                plannerUsed:
                    true,


                executionStarted:
                    true



            },



        plan:
            planResult.plan,



        planningContext:
            effectivePlanningContext,



        ...executionResult


    };


}
