/*
 * =========================================================
 * JESSICA LEARNING FACADE v2
 * =========================================================
 *
 * Единая асинхронная точка входа
 * Learning Layer.
 *
 *
 * Flow:
 *
 * Execution Trace
 *        ↓
 * Learning Facade
 *        ↓
 * Learning Coordinator
 *        ↓
 * Learning Queue
 *
 * =========================================================
 */


import {
    processLearning
} from "./learningCoordinator.js";


/*
 * =========================================================
 * EMPTY
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
            "Learning skipped",

        result:
            null

    };

}


/*
 * =========================================================
 * PROCESS EXECUTION LEARNING
 * =========================================================
 */


export async function processExecutionLearning(
    executionTrace
) {

    if(
        !executionTrace
    ){

        return buildEmptyLearningResult();

    }


    try {


        /*
         * processLearning является async.
         *
         * Раньше Promise ошибочно
         * обрабатывался как готовый Result.
         */


        const result =

            await processLearning(
                executionTrace
            );


        return {

            success:

                result?.success === true,

            queued:

                result?.queued === true,

            reason:

                result?.reason

                ||

                null,

            result:

                result

                ||

                null

        };


    }catch(error){


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

                error?.message

                ||

                "Learning execution error",

            result:
                null

        };

    }

}
