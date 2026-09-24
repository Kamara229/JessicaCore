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
    updateTraceFromResult,
    updateTraceFromSummary,
    finishExecutionTrace
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
 * User Task
 *      ↓
 * Execution Trace
 *      ↓
 * Task Decomposer
 *      ↓
 * Subtasks
 *      ↓
 * Subtask Runner
 *      ↓
 * Response Builder
 *      ↓
 * API Response
 *
 *
 * Ответственность:
 *
 * - принять задачу;
 * - создать execution trace;
 * - запустить декомпозицию;
 * - выбрать single / complex execution;
 * - вернуть результат.
 *
 *
 * НЕ содержит:
 *
 * - Planner;
 * - Experience;
 * - Tools;
 * - Validator;
 * - Composer;
 * - Learning;
 * - Storage;
 * - бизнес-логику.
 *
 * =========================================================
 */





/*
 * =========================================================
 * INVALID INPUT RESPONSE
 * =========================================================
 */


function buildInputError(
    executionTrace
) {


    return {

        success:false,


        stage:
            "input",


        text:
            "Задача не указана",


        executionTrace

    };


}





/*
 * =========================================================
 * DECOMPOSER ERROR
 * =========================================================
 */


function buildDecomposerError(
    executionTrace,
    text
) {


    finishExecutionTrace(
        executionTrace
    );


    return {

        success:false,


        stage:
            "decomposer",


        text:
            text ||
            "Jessica не смогла разобрать задачу",


        executionTrace

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
     * INPUT
     * =====================================================
     */


    if (
        !normalizedTask
    ) {

        return buildInputError(
            executionTrace
        );

    }







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
            "Jessica decomposer error:",
            error
        );


        return buildDecomposerError(
            executionTrace
        );


    }





    if (
        !decompositionResult?.success
    ) {


        return buildDecomposerError(

            executionTrace,

            decompositionResult?.text

        );


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


        return buildDecomposerError(

            executionTrace,

            "Jessica не обнаружила задач для выполнения"

        );


    }







    console.log(

        `Jessica decomposition: ${subtasks.length} subtask(s)`

    );









    /*
     * =====================================================
     * SINGLE TASK
     * =====================================================
     */


    if (
        subtasks.length === 1
    ) {


        let result;



        try {


            result =
                await executeSubtask(
                    subtasks[0]
                );


        } catch(error) {


            console.error(

                "Jessica single subtask error:",

                error

            );



            result = {


                id:
                    subtasks[0]?.id || null,


                status:
                    "FAILED",


                success:false,


                stage:
                    "subtask",


                result:
                    "Ошибка выполнения подзадачи"


            };


        }





        updateTraceFromResult(

            executionTrace,

            result

        );



        finishExecutionTrace(
            executionTrace
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


    let subtaskRunResult;



    try {


        subtaskRunResult =
            await runSubtasks(
                decomposition
            );



    } catch(error) {


        console.error(

            "Jessica complex execution error:",

            error

        );



        finishExecutionTrace(
            executionTrace
        );



        return {


            success:false,


            stage:
                "execution",


            text:
                "Ошибка выполнения сложной задачи",


            executionTrace


        };


    }







    /*
     * TRACE UPDATE
     */


    updateTraceFromSummary(

        executionTrace,

        subtaskRunResult

    );



    finishExecutionTrace(
        executionTrace
    );








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








    /*
     * =====================================================
     * RESPONSE
     * =====================================================
     */


    return await buildComplexTaskResponse(

        normalizedTask,

        decomposition,

        subtaskRunResult,

        executionTrace

    );



}
