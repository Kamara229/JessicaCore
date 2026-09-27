/*
 * =========================================================
 * JESSICA RESPONSE FACADE v1
 * =========================================================
 *
 * Единая точка формирования ответа.
 *
 *
 * Flow:
 *
 * Execution Result
 *        ↓
 * Response Facade
 *        ↓
 *
 * Single Response
 * Complex Response
 *
 *
 * Используется:
 *
 * - Jessica Core
 *
 *
 * НЕ:
 *
 * - выполняет задачи;
 * - анализирует ошибки;
 * - работает с Memory;
 * - изменяет Learning.
 *
 * =========================================================
 */


import {
    buildSingleTaskResponse
} from "./singleTaskResponse.js";


import {
    buildComplexTaskResponse
} from "./complexTaskResponse.js";









/*
 * =========================================================
 * DETECT RESPONSE TYPE
 * =========================================================
 */


function isComplexExecution(

    executionResult

)
{


    return Array.isArray(

        executionResult?.results

    );


}









/*
 * =========================================================
 * BUILD RESPONSE
 * =========================================================
 */


export async function buildJessicaResponse(

    {

        task,

        decomposition,

        executionResult,

        executionTrace

    }

)
{


    if(

        isComplexExecution(

            executionResult

        )

    )
    {


        return buildComplexTaskResponse(

            task,

            decomposition,

            executionResult,

            executionTrace

        );


    }







    return buildSingleTaskResponse(

        executionResult,

        decomposition,

        executionTrace

    );


}
