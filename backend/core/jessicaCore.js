import {
    decomposeTask
} from "./taskDecomposer.js";


import {
    executeSubtask,
    runSubtasks
} from "./subtaskRunner.js";


import {
    buildSingleTaskResponse
} from "./response/singleTaskResponse.js";


import {
    buildComplexTaskResponse
} from "./response/complexTaskResponse.js";


import {
    createExecutionTrace,
    updateTraceFromResult
} from "./trace/executionTrace.js";



/*
 * =========================================================
 * JESSICA CORE
 * =========================================================
 *
 * Центральный исполнитель Jessica.
 *
 *
 * Новый цикл:
 *
 * Task
 *   ↓
 * Task Decomposer
 *   ↓
 * Subtasks
 *   ↓
 * Subtask Runner
 *   ↓
 * Planner
 *   ↓
 * Tools
 *   ↓
 * Validator
 *   ↓
 * Response Builder
 *
 *
 * Этот файл НЕ содержит:
 *
 * - формирование ответа;
 * - Answer Composer;
 * - Complex Composer;
 * - Learning;
 * - Experience;
 * - Storage.
 *
 * =========================================================
 */



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

            success:
                false,

            stage:
                "input",

            text:
                "Задача не указана"

        };

    }



    /*
     * =====================================================
     * EXECUTION TRACE
     * =====================================================
     *
     * Подготавливаем контекст
     * для будущего Learning.
     *
     * =====================================================
     */


    const executionTrace =
        createExecutionTrace(
            normalizedTask
        );



    /*
     * =====================================================
     * 1. DECOMPOSE
     * =====================================================
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

            success:
                false,

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

            success:
                false,

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

            success:
                false,

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
     * 2. SINGLE TASK
     * =====================================================
     */


    if (
        subtasks.length === 1
    ) {


        const result =
            await executeSubtask(
                subtasks[0]
            );



        updateTraceFromResult(
            executionTrace,
            result
        );



        const response =
            buildSingleTaskResponse(
                result,
                decomposition
            );



        response.executionTrace =
            executionTrace;



        return response;

    }



    /*
     * =====================================================
     * 3. COMPLEX TASK
     * =====================================================
     *
     * Каждая подзадача выполняется независимо.
     *
     * =====================================================
     */


    const subtaskRunResult =
        await runSubtasks(
            decomposition
        );



    executionTrace.subtasks =
        subtaskRunResult.results || [];



    executionTrace.completed =
        subtaskRunResult.completed > 0;



    executionTrace.usedTools =
        executionTrace.subtasks
            .flatMap(
                item =>
                    item.usedTools || []
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
     * =====================================================
     * 4. COMPLEX RESPONSE
     * =====================================================
     */


    const response =
        await buildComplexTaskResponse(
            normalizedTask,
            decomposition,
            subtaskRunResult
        );



    response.executionTrace =
        executionTrace;



    return response;

}
