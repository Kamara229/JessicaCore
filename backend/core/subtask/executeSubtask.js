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
 * PlanningContext
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
 * - подготовить PlanningContext;
 * - создать первый план;
 * - передать управление Execution Cycle.
 *
 *
 * НЕ отвечает за:
 *
 * - выполнение инструментов;
 * - retry;
 * - replan;
 * - validation;
 * - обучение.
 *
 * =========================================================
 */



/*
 * =========================================================
 * EXECUTE ONE SUBTASK
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

        return {

            id:
                subtaskId,

            text:
                taskText,

            status:
                "FAILED",

            success:
                false,

            stage:
                "input",

            result:
                "Подзадача не содержит текста"

        };

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


    } catch(error) {


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



    /*
     * =====================================================
     * 2. PLANNING CONTEXT
     * =====================================================
     */


    const planningContext =
        experienceResult?.planningContext &&
        typeof experienceResult.planningContext === "object"

            ? experienceResult.planningContext

            : {};




    if (
        experienceResult?.found === true
    ) {


        console.log(

            "Jessica Experience selected:",

            {

                skillId:
                    planningContext
                        ?.experience
                        ?.skillId ||
                    null,


                version:
                    planningContext
                        ?.experience
                        ?.version ||
                    null,


                confidence:
                    experienceResult.confidence || 0

            }

        );


    } else {


        console.log(
            "Jessica Experience not found"
        );

    }




    /*
     * =====================================================
     * 3. INITIAL PLAN
     * =====================================================
     */


    let planResult;



    try {


        planResult =
            await createPlan(

                taskText,

                planningContext

            );


    } catch(error) {


        console.error(

            "Jessica Planner error:",

            error

        );


        return {

            id:
                subtaskId,

            text:
                taskText,

            success:
                false,

            status:
                "FAILED",

            stage:
                "planner",

            result:
                "Ошибка создания плана"

        };

    }




    if (
        !planResult?.success ||
        !planResult?.plan
    ) {


        return {

            id:
                subtaskId,


            text:
                taskText,


            success:
                false,


            status:
                "FAILED",


            stage:
                "planner",


            result:
                planResult?.text ||
                "План не создан"

        };

    }




    /*
     * =====================================================
     * 4. EXECUTION CYCLE
     * =====================================================
     *
     * ВАЖНО:
     *
     * Передаём Experience Context.
     *
     * Он используется:
     *
     * - Replanner;
     * - будущим Learning;
     * - анализом качества Skills.
     *
     * =====================================================
     */


    let executionResult;



    try {


        executionResult =
            await executePlanCycle(

                taskText,

                planResult.plan,

                planningContext

            );



    } catch(error) {


        console.error(

            "Jessica Execution Cycle error:",

            error

        );



        return {

            id:
                subtaskId,


            text:
                taskText,


            success:
                false,


            status:
                "FAILED",


            stage:
                "execution",


            result:
                "Ошибка цикла выполнения",


            plan:
                planResult.plan

        };

    }





    /*
     * =====================================================
     * 5. NORMALIZE RESULT
     * =====================================================
     */


    return {

        id:
            subtaskId,


        text:
            taskText,



        /*
         * Experience metadata.
         *
         * Не сохраняем полный Skill.
         *
         * Только ссылку:
         *
         * Learning
         * Statistics
         * Earnings
         *
         */


        experience: {


            found:
                experienceResult?.found === true,


            skillId:
                planningContext
                    ?.experience
                    ?.skillId ||
                null,


            version:
                planningContext
                    ?.experience
                    ?.version ||
                null,


            confidence:
                Number(
                    experienceResult?.confidence || 0
                )


        },



        /*
         * Полный результат Execution Cycle.
         */


        ...executionResult

    };

}
