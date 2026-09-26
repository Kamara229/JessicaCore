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


import {
    runLearningTrigger
} from "../experience/learning/learningTrigger.js";



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
 * Decomposer
 *      ↓
 * Subtasks
 *      ↓
 * Execution
 *      ↓
 * Response Builder
 *      ↓
 * Learning Trigger
 *      ↓
 * Learning Decision
 *
 *
 * НЕ содержит:
 *
 * - Planner;
 * - Tools;
 * - Validator;
 * - Experience logic;
 * - Storage;
 * - Skill creation.
 *
 * =========================================================
 */





/*
 * =========================================================
 * SAFE LEARNING
 * =========================================================
 */


function runLearningSafely(
    executionTrace
) {


    try {


        if (
            !executionTrace
        ) {

            return null;

        }


        return runLearningTrigger(
            executionTrace
        );


    } catch(error) {


        console.error(
            "Jessica Learning error:",
            error
        );


        return {

            triggered:false,

            reason:
                "Learning pipeline error"

        };

    }

}





/*
 * =========================================================
 * FAILED RESULT
 * =========================================================
 */


function buildFailureResponse(
{

    stage,

    text,

    executionTrace

}
) {


    finishExecutionTrace(
        executionTrace
    );


    return {

        success:false,

        stage,

        text,

        executionTrace

    };

}





/*
 * =========================================================
 * EXECUTE TASK
 * =========================================================
 */


export async function executeJessicaTask(
    task
) {


    const normalizedTask =
        typeof task === "string"
            ? task.trim()
            : "";



    const executionTrace =
        createExecutionTrace(
            normalizedTask
        );




    /*
     * INPUT
     */


    if (
        !normalizedTask
    ) {


        return buildFailureResponse({

            stage:
                "input",

            text:
                "Задача не указана",

            executionTrace

        });

    }






    /*
     * DECOMPOSE
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


        return buildFailureResponse({

            stage:
                "decomposer",

            text:
                "Jessica не смогла разобрать задачу",

            executionTrace

        });


    }




    if (
        !decompositionResult?.success
    ) {


        return buildFailureResponse({

            stage:
                "decomposer",

            text:
                decompositionResult?.text ||
                "Ошибка декомпозиции",

            executionTrace

        });

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


        return buildFailureResponse({

            stage:
                "decomposer",

            text:
                "Jessica не обнаружила задач",

            executionTrace

        });

    }





    console.log(
        `Jessica decomposition: ${subtasks.length}`
    );






    /*
     * =====================================================
     * SINGLE
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
                "Jessica single execution error:",
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




        const response =
            buildSingleTaskResponse(

                result,

                decomposition,

                executionTrace

            );



        /*
         * Запуск обучения
         */


        response.learning =
            runLearningSafely(
                executionTrace
            );



        return response;


    }







    /*
     * =====================================================
     * COMPLEX
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


        return buildFailureResponse({

            stage:
                "execution",

            text:
                "Ошибка выполнения сложной задачи",

            executionTrace

        });


    }







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
                subtaskRunResult.failed

        }

    );







    const response =
        await buildComplexTaskResponse(

            normalizedTask,

            decomposition,

            subtaskRunResult,

            executionTrace

        );





    /*
     * Автоматическое обучение
     */


    response.learning =
        runLearningSafely(
            executionTrace
        );




    return response;


}
