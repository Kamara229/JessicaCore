/*
 * =========================================================
 * JESSICA EXPERIENCE CONFIDENCE METRICS
 * =========================================================
 */


import {
    clampUnit,
    normalizePositiveInteger
} from "./confidenceNumbers.js";


export function calculateExperienceConfidence({

    matchScore = 0,

    successRate = 0,

    occurrences = 1

} = {}) {

    const normalizedMatch =

        clampUnit(
            matchScore
        );


    const normalizedSuccessRate =

        clampUnit(
            successRate
        );


    const normalizedOccurrences =

        Math.max(

            normalizePositiveInteger(
                occurrences
            ),

            1

        );


    const repeatScore =

        Math.min(

            normalizedOccurrences / 5,

            1

        );


    return Number(

        (

            normalizedMatch * 0.30

            +

            normalizedSuccessRate * 0.40

            +

            repeatScore * 0.30

        )
            .toFixed(2)

    );

}


export function calculateExperienceMaturity(
    occurrences
) {

    const count =

        Math.max(

            normalizePositiveInteger(
                occurrences
            ),

            1

        );


    const maturity =

        0.5

        +

        Math.min(
            count - 1,
            4
        )

        *

        0.125;


    return Number(

        Math.min(
            maturity,
            1
        )
            .toFixed(2)

    );

}


export function resolveExperienceMaturityLevel(
    occurrences
) {

    const count =

        Math.max(

            normalizePositiveInteger(
                occurrences
            ),

            1

        );


    if(
        count >= 5
    ){

        return "READY";

    }


    if(
        count >= 2
    ){

        return "LEARNING";

    }


    return "NEW";

}
