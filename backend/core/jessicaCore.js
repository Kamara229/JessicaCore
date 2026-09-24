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
 * Центральный координатор Jessica.
 *
 *
 * Flow:
 *
 * Task
 *  ↓
 * Decomposer
 *  ↓
 * Subtasks
 *  ↓
 * Subtask Runner
 *  ↓
 * Response Builder
 *
 *
 * Ответственность:
 *
 * - принять задачу;
 * - создать execution trace;
 * - запустить декомпозицию;
 * - выбрать single / complex flow;
 * - вернуть результат.
 *
 *
 * НЕ содержит:
 *
 * - Planner;
 * - Experience;
 * - Tools;
 * - Validation;
 * - Answer Composer;
 * - Complex Composer;
 * - Learning;
 * - Storage;
 * - бизнес-логику результата.
 *
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

            success:false,

            stage:"input",

            text:
                "Задача не указана"

        };

    }



    /*
     * =====================================================
     * TRACE
     * =====================================================
     */


    const executionTrace =
        createExecutionTrace(
            normalizedTask
        );




    /*
     * =====================================================
     * DECOMPOSE
     * =====================================================
     */


    let decompositionResult;


    try {


        decompositionResult =
            await decomposeTask(
                normalizedTask
            );


    } catch(error) {


        console.error(
            "Jessica Decomposer error:",
            error
        );


        return {

            success:false,

            stage:"decomposer",

            text:
                "Jessica не смогла разобрать задачу",

            executionTrace

        };

    }




    if (
        !decompositionResult?.success
    ) {


        return {

            success:false,

            stage:"decomposer",

            text:
                decompositionResult?.text ||
                "Jessica не смогла разобрать задачу",

            executionTrace

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

            success:false,

            stage:"decomposer",

            text:
                "Jessica не обнаружила задач",

            executionTrace

        };

    }



    console.log(
        `Jessica decomposition: ${subtasks.length}`
    );




    /*
     * =====================================================
     * SINGLE TASK
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



        return buildSingleTaskResponse(

            result,

            decomposition,

            executionTrace

        );

    }




    /*
     * =====================================================
     * COMPLEX TASK
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



    console.log(
        "Jessica complex result:",
        {

            total:
                subtaskRunResult.total,

            completed:
                subtaskRunResult.completed,

            failed:
                subtaskRunResult.failed,

            clarification:
                subtaskRunResult.needsClarification

        }
    );



    return await buildComplexTaskResponse(

        normalizedTask,

        decomposition,

        subtaskRunResult,

        executionTrace

    );

}
