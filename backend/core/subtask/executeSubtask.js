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
 * Выполняет одну подзадачу.
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
 * - получить текст подзадачи;
 * - найти подходящий Experience;
 * - получить PlanningContext;
 * - создать первоначальный план;
 * - передать план и контекст в Execution Cycle.
 *
 *
 * НЕ отвечает за:
 *
 * - выполнение инструментов;
 * - retry;
 * - replan;
 * - validation;
 * - Learning.
 *
 * =========================================================
 */


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



    /*
     * =====================================================
     * 2. INITIAL PLANNING CONTEXT
     * =====================================================
     */


    const initialPlanningContext =
        experienceResult?.planningContext &&
        typeof experienceResult.planningContext === "object"

            ? experienceResult.planningContext

            : {};



    /*
     * =====================================================
     * EXPERIENCE LOG
     * =====================================================
     */


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
                        experienceResult?.confidence || 0
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
     * 3. INITIAL PLAN
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
     * 4. EFFECTIVE PLANNING CONTEXT
     * =====================================================
     *
     * Planner может нормализовать или дополнить контекст.
     *
     * Поэтому после createPlan используем context,
     * возвращённый Planner.
     *
     * Если его нет — сохраняем первоначальный.
     *
     * =====================================================
     */


    const effectivePlanningContext =
        planResult?.context &&
        typeof planResult.context === "object"

            ? planResult.context

            : initialPlanningContext;



    /*
     * =====================================================
     * 5. EXECUTION CYCLE
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
                planResult.plan,

            planningContext:
                effectivePlanningContext

        };

    }



    /*
     * =====================================================
     * 6. RESULT
     * =====================================================
     */


    return {

        id:
            subtaskId,

        text:
            taskText,


        /*
         * Краткая ссылка на использованный Experience.
         *
         * Полный Skill сюда не копируем.
         */


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


        /*
         * Результат Execution Cycle.
         */


        ...executionResult

    };

}
