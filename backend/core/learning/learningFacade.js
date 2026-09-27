/*
 * =========================================================
 * JESSICA LEARNING FACADE v1
 * =========================================================
 *
 * Единая точка входа Learning Layer.
 *
 *
 * Flow:
 *
 * Execution Result
 *        ↓
 * Learning Facade
 *        ↓
 * Learning Coordinator
 *        ↓
 * Learning Queue
 *
 *
 * Используется:
 *
 * - Jessica Core
 * - Execution Layer
 *
 *
 * НЕ:
 *
 * - создаёт Skills;
 * - изменяет Experience;
 * - хранит память;
 * - принимает решение обучения.
 *
 * =========================================================
 */


import {
    processLearning
} from "./learningCoordinator.js";









/*
 * =========================================================
 * EMPTY RESULT
 * =========================================================
 */


function buildEmptyLearningResult()
{


    return {


        success:
            false,


        queued:
            false,


        reason:
            "Learning skipped"


    };


}









/*
 * =========================================================
 * PROCESS EXECUTION LEARNING
 * =========================================================
 */


export function processExecutionLearning(

    executionTrace

)
{


    if(
        !executionTrace
    )
    {


        return buildEmptyLearningResult();

    }







    try
    {


        const result =

            processLearning(

                executionTrace

            );





        return {


            success:

                result?.success === true,



            queued:

                result?.queued === true,



            result:

                result || null



        };



    }

    catch(error)
    {


        console.error(

            "Jessica Learning Facade error:",

            error

        );




        return {


            success:
                false,


            queued:
                false,


            reason:
                "Learning execution error"


        };


    }


}
