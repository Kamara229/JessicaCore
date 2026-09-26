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
    processLearning
} from "./learning/learningCoordinator.js";


import {
    retrieveRelevantExperience
} from "./experience/memoryRetriever.js";


import {
    buildMemoryContext
} from "./memory/memoryContextBuilder.js";




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
 * Memory Retrieval
 *      ↓
 * Memory Context
 *      ↓
 * Execution Trace
 *      ↓
 * Task Decomposer
 *      ↓
 * Subtasks
 *      ↓
 * Execution
 *      ↓
 * Response Builder
 *      ↓
 * Learning Coordinator
 *      ↓
 * Learning Queue
 *
 *
 * НЕ содержит:
 *
 * - Planner;
 * - Tools;
 * - Validator;
 * - Experience Storage;
 * - Skill creation.
 *
 * =========================================================
 */





/*
 * =========================================================
 * FAILURE RESPONSE
 * =========================================================
 */


function buildFailureResponse({

    stage,

    text,

    executionTrace

}) {


    finishExecutionTrace(
        executionTrace
    );


    return {

        success:
            false,

        stage,

        text,

        executionTrace

    };

}





/*
 * =========================================================
 * MEMORY LOAD
 * =========================================================
 */


async function loadMemoryContext(
    task
) {


    try {


        const experience =
            await retrieveRelevantExperience(
                task
            );



        return buildMemoryContext(

            experience?.skills || []

        );



    } catch(error) {


        console.error(

            "Jessica Memory Retrieval error:",

            error

        );


        return {


            hasExperience:
                false,


            skills:
                [],


            skillCount:
                0,


            summary:
                []


        };


    }


}





/*
 * =========================================================
 * SAFE LEARNING
 * =========================================================
 */


function attachLearning(
    response,
    executionTrace
) {


    try {


        response.learning =
            processLearning(
                executionTrace
            );


    } catch(error) {


        console.error(

            "Jessica Learning Coordinator error:",

            error

        );



        response.learning = {


            success:
                false,


            queued:
                false,


            reason:
                "Learning coordinator error"


        };


    }



    return response;


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
     * =====================================================
     * INPUT
     * =====================================================
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
     * =====================================================
     * MEMORY RETRIEVAL
     * =====================================================
     */


    const memoryContext =
        await loadMemoryContext(
            normalizedTask
        );



    executionTrace.memory =
        memoryContext;






    /*
     * =====================================================
     * DECOMPOSE
     * =====================================================
     */


    let decompositionResult;



    try {


        decompositionResult =
            await decomposeTask({

                task:
                    normalizedTask,


                memoryContext

            });



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

                "Jessica single execution error:",

                error

            );



            result = {


                id:
                    subtasks[0]?.id || null,


                status:
                    "FAILED",


                success:
                    false,


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





        return attachLearning(

            response,

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






    const response =
        await buildComplexTaskResponse(

            normalizedTask,

            decomposition,

            subtaskRunResult,

            executionTrace

        );





    return attachLearning(

        response,

        executionTrace

    );


}
