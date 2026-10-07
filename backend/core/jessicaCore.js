/*
 * =========================================================
 * JESSICA CORE v3
 * =========================================================
 *
 * Центральный координатор Jessica.
 *
 *
 * Flow:
 *
 * User Task
 *      ↓
 * Memory
 *      ↓
 * Decomposition
 *      ↓
 * Execution Router
 *      ↓
 * Outer Execution Trace
 *      ↓
 * Response
 *      ↓
 * Learning
 *
 *
 * НЕ:
 *
 * - выполняет Tools;
 * - создаёт Plan;
 * - валидирует Result;
 * - хранит Experience;
 * - создаёт Skills.
 *
 * =========================================================
 */


import {
    decomposeTask
} from "./taskDecomposer.js";


import {
    buildExecutionMemoryContext
} from "./memory/executionMemoryFacade.js";


import {
    createJessicaExecutionTrace,
    recordExecutionResult,
    recordExecutionSummary,
    completeExecutionTrace
} from "./trace/executionTraceFacade.js";


import {
    executeByRoute
} from "./execution/executionRouter.js";


import {
    buildJessicaResponse
} from "./response/responseFacade.js";


import {
    processExecutionLearning
} from "./learning/learningFacade.js";


/*
 * =========================================================
 * FAILURE
 * =========================================================
 */


function buildCoreFailure(

    stage,

    text,

    trace

) {

    completeExecutionTrace(
        trace
    );


    return {

        success:
            false,

        stage,

        text,

        executionTrace:
            trace

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

        createJessicaExecutionTrace(
            normalizedTask
        );


    /*
     * =====================================================
     * INPUT
     * =====================================================
     */


    if(
        !normalizedTask
    ){

        return buildCoreFailure(

            "input",

            "Задача не указана",

            executionTrace

        );

    }


    /*
     * =====================================================
     * MEMORY
     * =====================================================
     */


    const memoryResult =

        await buildExecutionMemoryContext(
            normalizedTask
        );


    /*
     * =====================================================
     * DECOMPOSITION
     * =====================================================
     */


    let decompositionResult;


    try {


        decompositionResult =

            await decomposeTask({

                task:
                    normalizedTask,

                memoryContext:
                    memoryResult.memoryContext

            });


    }catch(error){


        console.error(

            "Jessica decomposer error:",

            error

        );


        return buildCoreFailure(

            "decomposer",

            "Jessica не смогла разобрать задачу",

            executionTrace

        );

    }


    if(
        !decompositionResult?.success
    ){

        return buildCoreFailure(

            "decomposer",

            decompositionResult?.text

            ||

            "Ошибка декомпозиции",

            executionTrace

        );

    }


    const decomposition =

        decompositionResult.decomposition;


    /*
     * =====================================================
     * EXECUTION
     * =====================================================
     */


    let executionResult;


    try {


        executionResult =

            await executeByRoute(
                decomposition
            );


    }catch(error){


        console.error(

            "Jessica execution router error:",

            error

        );


        return buildCoreFailure(

            "execution",

            "Ошибка выполнения задачи",

            executionTrace

        );

    }


    /*
     * =====================================================
     * OUTER TRACE UPDATE
     * =====================================================
     */


    if(
        Array.isArray(
            executionResult?.results
        )
    ){

        recordExecutionSummary(

            executionTrace,

            executionResult

        );

    }else{

        recordExecutionResult(

            executionTrace,

            executionResult

        );

    }


    /*
     * Trace должен быть полностью
     * завершён ДО Learning.
     */


    completeExecutionTrace(
        executionTrace
    );


    /*
     * =====================================================
     * RESPONSE
     * =====================================================
     */


    const response =

        await buildJessicaResponse({

            task:
                normalizedTask,

            decomposition,

            executionResult,

            executionTrace

        });


    /*
     * =====================================================
     * LEARNING
     * =====================================================
     *
     * Раньше Learning Promise
     * не ожидался.
     *
     * Теперь Queue persistence
     * завершается до возврата Response.
     *
     * =====================================================
     */


    const learningResult =

        await processExecutionLearning(
            executionTrace
        );


    response.learning =
        learningResult;


    /*
     * Диагностический лог.
     *
     * Не содержит полного Candidate
     * или пользовательского контента.
     */


    console.log(

        "Jessica Learning result:",

        {

            success:
                learningResult?.success === true,

            queued:
                learningResult?.queued === true,

            reason:
                learningResult?.reason || null,

            traceStatus:
                executionTrace?.status || null,

            completed:

                executionTrace
                    ?.statistics
                    ?.completed

                ??

                executionTrace
                    ?.stats
                    ?.completed

                ??

                0,

            experienceUsed:

                executionTrace
                    ?.experienceUsage
                    ?.used === true,

            experienceSkills:

                Array.isArray(
                    executionTrace
                        ?.experienceUsage
                        ?.skills
                )

                    ? executionTrace
                        .experienceUsage
                        .skills
                        .length

                    : 0

        }

    );


    return response;

}
