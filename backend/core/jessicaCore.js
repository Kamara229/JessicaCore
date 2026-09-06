import {
    decomposeTask
} from "./taskDecomposer.js";

import {
    executeSubtask,
    runSubtasks
} from "./subtaskRunner.js";

import {
    composeComplexAnswer
} from "./complexAnswerComposer.js";


/*
 * =========================================================
 * JESSICA CORE
 * =========================================================
 *
 * Центральный исполнитель Jessica.
 *
 * Новый цикл:
 *
 * Task
 *   ↓
 * Task Decomposer
 *   ↓
 * одна или несколько подзадач
 *   ↓
 * Subtask Runner
 *   ↓
 * Planner
 *   ↓
 * TaskRunner
 *   ↓
 * Tools
 *   ↓
 * Answer Composer
 *   ↓
 * Validator
 *   ↓
 * Complex Answer Composer
 *
 *
 * Главное изменение:
 *
 * ошибка одной подзадачи больше
 * не должна останавливать остальные.
 */


/*
 * =========================================================
 * SINGLE SUBTASK RESULT
 * =========================================================
 *
 * Преобразует результат одной подзадачи
 * в обычный ответ /api/solve.
 */


function buildSingleTaskResponse(
    result,
    decomposition
) {

    /*
     * -----------------------------------------------------
     * SUCCESS
     * -----------------------------------------------------
     */


    if (
        result.status ===
        "COMPLETED"
    ) {

        return {

            success: true,

            text:
                result.result,

            engine:
                "jessica-core",

            mode:
                "single",

            validated:
                result.validated === true,

            answerSource:
                result.answerSource || "unknown",

            usedTools:
                result.usedTools || [],

            decomposition,

            plan:
                result.plan || null,

            toolResults:
                result.toolResults || []

        };

    }


    /*
     * -----------------------------------------------------
     * NEEDS CLARIFICATION
     * -----------------------------------------------------
     */


    if (
        result.status ===
        "NEEDS_CLARIFICATION"
    ) {

        return {

            success: false,

            needsClarification:
                true,

            text:
                result.result ||
                "Для выполнения задачи требуется уточнение.",

            engine:
                "jessica-core",

            mode:
                "single",

            stage:
                result.stage || "subtask",

            decomposition,

            plan:
                result.plan || null,

            toolResults:
                result.toolResults || []

        };

    }


    /*
     * -----------------------------------------------------
     * FAILED
     * -----------------------------------------------------
     */


    return {

        success: false,

        shouldRetry:
            result.shouldRetry === true,

        text:
            result.result ||
            "Jessica не смогла выполнить задачу.",

        engine:
            "jessica-core",

        mode:
            "single",

        stage:
            result.stage || "subtask",

        decomposition,

        plan:
            result.plan || null,

        toolResults:
            result.toolResults || []

    };

}


/*
 * =========================================================
 * COMPLEX TASK RESULT
 * =========================================================
 */


async function buildComplexTaskResponse(
    originalTask,
    decomposition,
    subtaskRunResult
) {

    /*
     * Формируем единый пользовательский ответ.
     */
    const composed =
        await composeComplexAnswer(
            originalTask,
            decomposition,
            subtaskRunResult
        );


    const completed =
        subtaskRunResult.completed || 0;


    const needsClarification =
        subtaskRunResult.needsClarification || 0;


    const failed =
        subtaskRunResult.failed || 0;


    const total =
        subtaskRunResult.total || 0;


    /*
     * Частичный результат:
     *
     * хотя бы одна подзадача выполнена,
     * но некоторые требуют уточнения
     * или завершились ошибкой.
     */
    const partial =
        completed > 0 &&
        (
            needsClarification > 0 ||
            failed > 0
        );


    /*
     * =====================================================
     * ЕСТЬ ХОТЯ БЫ ОДИН ПОЛЕЗНЫЙ РЕЗУЛЬТАТ
     * =====================================================
     *
     * Не делаем всю задачу FAILED.
     *
     * Пользователь получает всё,
     * что Jessica смогла выполнить.
     */


    if (
        completed > 0
    ) {

        return {

            success: true,

            text:
                composed.text,

            engine:
                "jessica-core",

            mode:
                "complex",

            partial,

            summary: {

                total,

                completed,

                needsClarification,

                failed

            },

            decomposition,

            subtasks:
                subtaskRunResult.results || [],

            answerSource:
                composed.source || "unknown"

        };

    }


    /*
     * =====================================================
     * НИ ОДНА ПОДЗАДАЧА НЕ ВЫПОЛНЕНА
     * =====================================================
     */


    if (
        needsClarification > 0 &&
        failed === 0
    ) {

        return {

            success: false,

            needsClarification:
                true,

            text:
                composed.text,

            engine:
                "jessica-core",

            mode:
                "complex",

            summary: {

                total,

                completed,

                needsClarification,

                failed

            },

            decomposition,

            subtasks:
                subtaskRunResult.results || []

        };

    }


    /*
     * Есть одновременно ошибки
     * и запросы на уточнение,
     * но нет успешных результатов.
     */


    return {

        success: false,

        needsClarification:
            needsClarification > 0,

        text:
            composed.text,

        engine:
            "jessica-core",

        mode:
            "complex",

        summary: {

            total,

            completed,

            needsClarification,

            failed

        },

        decomposition,

        subtasks:
            subtaskRunResult.results || []

    };

}


/*
 * =========================================================
 * EXECUTE JESSICA TASK
 * =========================================================
 */


export async function executeJessicaTask(
    task
) {

    const normalizedTask =
        typeof task === "string"
            ? task.trim()
            : "";


    if (!normalizedTask) {

        return {

            success: false,

            stage:
                "input",

            text:
                "Задача не указана"

        };

    }


    /*
     * -----------------------------------------------------
     * 1. DECOMPOSE
     * -----------------------------------------------------
     */


    let decompositionResult;


    try {

        decompositionResult =
            await decomposeTask(
                normalizedTask
            );

    } catch (error) {

        console.error(
            "Jessica Decomposer exception:",
            error
        );


        return {

            success: false,

            stage:
                "decomposer",

            text:
                "Jessica не смогла разобрать задачу"

        };

    }


    if (
        !decompositionResult?.success
    ) {

        return {

            success: false,

            stage:
                "decomposer",

            text:
                decompositionResult?.text ||
                "Jessica не смогла разобрать задачу"

        };

    }


    const decomposition =
        decompositionResult.decomposition;


    const subtasks =
        Array.isArray(
            decomposition?.subtasks
        )
            ? decomposition.subtasks
            : [];


    if (
        subtasks.length === 0
    ) {

        return {

            success: false,

            stage:
                "decomposer",

            text:
                "Jessica не обнаружила задач для выполнения"

        };

    }


    console.log(
        `Jessica decomposition: ${subtasks.length} subtask(s)`
    );


    console.log(
        JSON.stringify(
            decomposition
        )
    );


    /*
     * =====================================================
     * 2. SIMPLE TASK
     * =====================================================
     *
     * Если задача одна,
     * не запускаем Complex Composer.
     *
     * Она идёт через обычный маршрут:
     *
     * Planner
     * → Runner
     * → Tools
     * → Answer Composer
     * → Validator
     */


    if (
        subtasks.length === 1
    ) {

        const result =
            await executeSubtask(
                subtasks[0]
            );


        return buildSingleTaskResponse(
            result,
            decomposition
        );

    }


    /*
     * =====================================================
     * 3. COMPLEX TASK
     * =====================================================
     *
     * Каждая подзадача выполняется независимо.
     *
     * Ошибка одной подзадачи
     * НЕ останавливает остальные.
     */


    const subtaskRunResult =
        await runSubtasks(
            decomposition
        );


    console.log(
        "Jessica complex task result:",
        JSON.stringify({
            total:
                subtaskRunResult.total,

            completed:
                subtaskRunResult.completed,

            needsClarification:
                subtaskRunResult.needsClarification,

            failed:
                subtaskRunResult.failed
        })
    );


    /*
     * -----------------------------------------------------
     * 4. FINAL COMPLEX ANSWER
     * -----------------------------------------------------
     */


    return await buildComplexTaskResponse(
        normalizedTask,
        decomposition,
        subtaskRunResult
    );

}
