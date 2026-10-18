/*
 * =========================================================
 * JESSICA EXPERIENCE OCCURRENCES
 * =========================================================
 */


import {
    normalizePositiveInteger
} from "./confidenceNumbers.js";


export function getOccurrences(
    trace
) {

    const explicit =

        normalizePositiveInteger(
            trace?.experienceOccurrences
        );


    if(
        explicit > 0
    ){

        return explicit;

    }


    const traceOccurrences =

        normalizePositiveInteger(

            trace?.stats?.occurrences

            ??

            trace?.statistics?.occurrences

        );


    if(
        traceOccurrences > 0
    ){

        return traceOccurrences;

    }


    if(
        Array.isArray(
            trace?.examples
        )
        &&
        trace.examples.length > 1
    ){

        return trace.examples.length;

    }


    const usedSkills =

        Array.isArray(
            trace
                ?.experienceUsage
                ?.skills
        )

            ? trace.experienceUsage.skills

            : [];


    const usedSkill =

        usedSkills[0]

        ||

        null;


    if(usedSkill){

        const storedOccurrences =

            normalizePositiveInteger(

                usedSkill
                    ?.learning
                    ?.occurrences

                ??

                usedSkill?.occurrences

            );


        if(
            storedOccurrences > 0
        ){

            return storedOccurrences + 1;

        }


        const runtimeOccurrences =

            normalizePositiveInteger(

                usedSkill
                    ?.statistics
                    ?.successfulRuns

            )

            +

            normalizePositiveInteger(

                usedSkill
                    ?.statistics
                    ?.failedRuns

            );


        if(
            runtimeOccurrences > 0
        ){

            return runtimeOccurrences + 1;

        }

    }


    return 1;

}
