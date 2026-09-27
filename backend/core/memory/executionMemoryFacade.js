/*
 * =========================================================
 * JESSICA EXECUTION MEMORY FACADE v1
 * =========================================================
 *
 * Единая точка доступа Execution к Memory Layer.
 *
 *
 * Flow:
 *
 * Task
 *   ↓
 * Memory Facade
 *   ↓
 * Experience Retrieval
 *   ↓
 * Memory Context
 *
 *
 * Используется:
 *
 * - Jessica Core
 * - Execution Context Builder
 * - Planning Layer
 *
 *
 * НЕ:
 *
 * - хранит память;
 * - создаёт Skills;
 * - изменяет Experience;
 * - обучает систему.
 *
 * =========================================================
 */


import {
    retrieveRelevantExperience
} from "../experience/memoryRetriever.js";


import {
    buildMemoryContext
} from "./memoryContextBuilder.js";









/*
 * =========================================================
 * EMPTY CONTEXT
 * =========================================================
 */


function buildEmptyMemoryContext()
{


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









/*
 * =========================================================
 * LOAD MEMORY
 * =========================================================
 */


export async function buildExecutionMemoryContext(

    task

) {


    try {


        const experience =

            await retrieveRelevantExperience(

                task

            );






        return {


            experience:

                experience || null,



            memoryContext:

                buildMemoryContext(

                    experience?.skills || []

                )


        };



    }

    catch(error)
    {


        console.error(

            "Jessica Execution Memory error:",

            error

        );




        return {


            experience:

                null,



            memoryContext:

                buildEmptyMemoryContext()


        };


    }


}









/*
 * =========================================================
 * HELPERS
 * =========================================================
 */


export function hasExecutionExperience(

    memory

)
{


    return (

        memory?.experience?.found === true

    );


}
