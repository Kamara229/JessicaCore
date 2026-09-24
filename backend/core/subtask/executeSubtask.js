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
 *    ↓
 * Experience
 *    ↓
 * Planning Context
 *    ↓
 * Planner
 *    ↓
 * Execution Cycle
 *    ↓
 * Result
 *
 *
 * Ответственность:
 *
 * - найти Experience;
 * - создать план;
 * - запустить Execution Cycle;
 * - вернуть полный результат.
 *
 *
 * НЕ отвечает за:
 *
 * - выполнение нескольких задач;
 * - общий ответ;
 * - Learning;
 * - Storage.
 *
 * =========================================================
 */



/*
 * =========================================================
 * FAILED RESULT BUILDER
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
 * EXECUTE SUBTASK
 * =========================================================
 */


export async function executeSubtask(
    subtask
) {


    /*
     * =====================================================
     * INPUT
     * =====================================================
     */


    const subtaskId =
        subtask?.id ?? null;


    const taskText =
        typeof subtask?.text === "string"
            ? subtask.text.trim()
            : "";



    if (!taskText) {


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
     * 1. EXPERIENCE
     * =====================================================
     */


    let experienceResult;


    try {


        experienceResult =
            await resolveExperience(
                taskText
            );


    } catch (error) {


        console.error(
            `Jessica Experience error ${subtaskId}:`,
            error
        );


        experienceResult = {

            found:
                false,

            confidence:
                0,

            planningContext:
                {}

        };

    }



    const initialPlanningContext =
        experienceResult?.planningContext &&
        typeof experienceResult.planningContext === "object"

            ? experienceResult.planningContext

            : {};



    if (
        experienceResult?.found === true
    ) {


        console.log(
            `Subtask ${subtaskId} Experience found:`,
            {

                skillId:
                    initialPlanningContext
                        ?.experience
                        ?.skillId ||
                    null,

                version:
                    initialPlanningContext
                        ?.experience
                        ?.version ||
                    null,

                matchConfidence:
                    Number(
                        experienceResult.confidence || 0
                    )

            }
        );


    } else {


        console.log(
            `Subtask ${subtaskId}: Experience not found`
        );


    }




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


    } catch (error) {


        console.error(
            `Subtask ${subtaskId} Planner error:`,
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
        planResult?.context &&
        typeof planResult.context === "object"

            ? planResult.context

            : initialPlanningContext;




    /*
     * =====================================================
     * 4. EXECUTION CYCLE
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


    } catch (error) {


        console.error(
            `Subtask ${subtaskId} Execution Cycle error:`,
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



        experience: {

            found:
                experienceResult?.found === true,


            skillId:
                effectivePlanningContext
                    ?.experience
                    ?.skillId ||
                null,


            version:
                effectivePlanningContext
                    ?.experience
                    ?.version ||
                null,


            matchConfidence:
                Number(
                    experienceResult?.confidence || 0
                )

        },



        executionMeta: {

            experienceUsed:
                experienceResult?.found === true,


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
