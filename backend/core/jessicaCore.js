/*
 * =========================================================
 * JESSICA CORE v2
 * =========================================================
 *
 * Центральный координатор Jessica.
 *
 *
 * Flow:
 *
 * User Task
 *      ↓
 * Memory Facade
 *      ↓
 * Task Decomposer
 *      ↓
 * Execution Router
 *      ↓
 * Trace Facade
 *      ↓
 * Response Facade
 *      ↓
 * Learning Facade
 *
 *
 * НЕ:
 *
 * - выполняет Tools;
 * - создаёт Plan;
 * - валидирует результат;
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

)
{


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

)
{


    const normalizedTask =

        typeof task === "string"

            ?

            task.trim()

            :

            "";






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
    )
    {


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



    try
    {


        decompositionResult =

            await decomposeTask({

                task:

                    normalizedTask,


                memoryContext:

                    memoryResult.memoryContext


            });


    }

    catch(error)
    {


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
    )
    {


        return buildCoreFailure(

            "decomposer",

            decompositionResult?.text ||

            "Ошибка декомпозиции",

            executionTrace

        );


    }








    const decomposition =

        decompositionResult.decomposition;









    /*
     * =====================================================
     * EXECUTION ROUTER
     * =====================================================
     */


    let executionResult;



    try
    {


        executionResult =

            await executeByRoute(

                decomposition

            );


    }

    catch(error)
    {


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
     * TRACE UPDATE
     * =====================================================
     */


    if(

        Array.isArray(

            executionResult?.results

        )

    )
    {


        recordExecutionSummary(

            executionTrace,

            executionResult

        );


    }

    else
    {


        recordExecutionResult(

            executionTrace,

            executionResult

        );


    }








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
     */


    response.learning =

        processExecutionLearning(

            executionTrace

        );









    return response;


}
