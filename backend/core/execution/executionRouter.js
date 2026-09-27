/*
 * =========================================================
 * JESSICA EXECUTION ROUTER v1
 * =========================================================
 *
 * Маршрутизатор выполнения задач.
 *
 *
 * Flow:
 *
 * Decomposition
 *        ↓
 * Execution Router
 *        ↓
 *
 * Single Task Executor
 *        |
 *        |
 * Complex Task Executor
 *
 *
 * НЕ:
 *
 * - создаёт план;
 * - валидирует результат;
 * - сохраняет Learning;
 * - управляет Trace.
 *
 * =========================================================
 */


import {
    executeSubtask,
    runSubtasks
} from "../subtaskRunner.js";









/*
 * =========================================================
 * NORMALIZE SUBTASKS
 * =========================================================
 */


function getSubtasks(

    decomposition

)
{


    return Array.isArray(
        decomposition?.subtasks
    )

        ?

        decomposition.subtasks

        :

        [];

}









/*
 * =========================================================
 * SINGLE EXECUTION
 * =========================================================
 */


async function executeSingle(

    subtasks

)
{


    return executeSubtask(

        subtasks[0]

    );


}









/*
 * =========================================================
 * COMPLEX EXECUTION
 * =========================================================
 */


async function executeComplex(

    decomposition

)
{


    return runSubtasks(

        decomposition

    );


}









/*
 * =========================================================
 * MAIN ROUTER
 * =========================================================
 */


export async function executeByRoute(

    decomposition

)
{


    const subtasks =

        getSubtasks(

            decomposition

        );







    if(
        subtasks.length === 0
    )
    {


        return {


            success:
                false,


            stage:
                "execution-router",


            text:
                "Нет задач для выполнения"


        };

    }








    /*
     * Single task
     */


    if(
        subtasks.length === 1
    )
    {


        return executeSingle(

            subtasks

        );


    }









    /*
     * Complex task
     */


    return executeComplex(

        decomposition

    );


}
