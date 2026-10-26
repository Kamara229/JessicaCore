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
import {hasExperienceStorageIntegrityFailure} from "./experienceStorageIntegrityGate.js";


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


    if(hasExperienceStorageIntegrityFailure(executionTrace)){
        return {
            success:false,
            queued:false,
            skipped:true,
            reason:"experience_storage_integrity_failure",
            result:null
        };
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
