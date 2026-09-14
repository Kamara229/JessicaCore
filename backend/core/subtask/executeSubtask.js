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
 * Рабочая цепочка:
 *
 * Subtask
 *   ↓
 * Experience
 *   ↓
 * PlanningContext
 *   ↓
 * Planner
 *   ↓
 * Execution Cycle
 *
 *
 * Если подходящий Experience не найден
 * или Experience временно недоступен:
 *
 * PlanningContext остаётся пустым
 * и Planner работает обычным способом.
 *
 *
 * Этот файл НЕ:
 *
 * - запускает несколько подзадач;
 * - считает общую статистику;
 * - хранит Experience;
 * - содержит алгоритм поиска Experience;
 * - содержит внутреннюю реализацию Planner.
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
     * PREPARE INPUT
     * =====================================================
     */


    const subtaskId =
        subtask?.id ?? null;


    const taskText =
        typeof subtask?.text === "string"
            ? subtask.text.trim()
            : "";


    /*
     * =====================================================
     * INPUT VALIDATION
     * =====================================================
     */


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
     *
     * Первый слой поиска решения Jessica.
     *
     * Experience Core:
     *
     * - загружает активные Skills из Storage;
     * - ищет подходящий Skill;
     * - формирует контекст для Planner.
     *
     * =====================================================
     */


    let experienceResult;


    try {


        experienceResult =
            await resolveExperience(
                taskText
            );


    } catch (error) {


        /*
         * Это дополнительная страховка.
         *
         * Experience Core уже сам должен
         * обрабатывать свои ошибки.
         *
         * Даже если что-то пошло не так,
         * выполнение задачи продолжается.
         */


        console.error(
            `Subtask ${subtaskId} Experience exception:`,
            error
        );


        experienceResult = {

            found:
                false,

            experience:
                null,

            confidence:
                0,

            planningContext: {

                experience:
                    null,

                metadata:
                    {}

            }

        };


    }


    /*
     * =====================================================
     * PLANNING CONTEXT
     * =====================================================
     */


    const planningContext =
        experienceResult?.planningContext &&
        typeof experienceResult.planningContext === "object"
            ? experienceResult.planningContext
            : {};


    /*
     * =====================================================
     * EXPERIENCE LOG
     * =====================================================
     *
     * В лог выводим только служебную информацию.
     *
     * Содержимое всего Skill здесь не печатаем.
     * =====================================================
     */


    if (
        experienceResult?.found === true
    ) {


        console.log(
            `Subtask ${subtaskId} Experience found:`,
            {
                skillId:
                    planningContext
                        ?.experience
                        ?.skillId || null,

                version:
                    planningContext
                        ?.experience
                        ?.version || null,

                matchConfidence:
                    experienceResult.confidence || 0
            }
        );


    } else {


        console.log(
            `Subtask ${subtaskId}: Experience not found`
        );


    }


    /*
     * =====================================================
     * 2. CREATE INITIAL PLAN
     * =====================================================
     */


    let planResult;


    try {


        planResult =
            await createPlan(

                taskText,

                planningContext

            );


    } catch (error) {


        console.error(
            `Subtask ${subtaskId} planner exception:`,
            error
        );


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
                "planner",

            result:
                "Не удалось построить план"

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

            status:
                "FAILED",

            success:
                false,

            stage:
                "planner",

            result:
                planResult?.text ||
                "Не удалось построить план"

        };

    }


    /*
     * =====================================================
     * 3. EXECUTION CYCLE
     * =====================================================
     *
     * План передаётся существующему циклу:
     *
     * run
     * → compose
     * → validate
     * → replan
     * → retry
     *
     * =====================================================
     */


    let executionResult;


    try {


        executionResult =
            await executePlanCycle(

                taskText,

                planResult.plan

            );


    } catch (error) {


        console.error(
            `Subtask ${subtaskId} execution cycle exception:`,
            error
        );


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
                "execution",

            result:
                "Непредвиденная ошибка цикла выполнения",

            plan:
                planResult.plan

        };


    }


    /*
     * =====================================================
     * 4. NORMALIZE RESULT
     * =====================================================
     */


    return {

        id:
            subtaskId,

        text:
            taskText,


        /*
         * Сохраняем краткую информацию
         * о применённом Experience.
         *
         * Она позже пригодится:
         *
         * - Learning;
         * - статистике;
         * - Earnings;
         * - анализу качества Skills.
         */


        experience: {

            found:
                experienceResult?.found === true,

            skillId:
                planningContext
                    ?.experience
                    ?.skillId || null,

            version:
                planningContext
                    ?.experience
                    ?.version || null,

            matchConfidence:
                Number(
                    experienceResult?.confidence || 0
                )

        },


        ...executionResult

    };


}
