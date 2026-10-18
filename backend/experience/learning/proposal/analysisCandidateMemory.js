/*
 * =========================================================
 * JESSICA PROPOSAL ANALYSIS CANDIDATE MEMORY
 * =========================================================
 */

import {
    isObject,
    normalizeText
} from "./proposalUtils.js";


function optionalNumber(
    value
) {

    if(
        value === undefined
        ||
        value === null
    ){

        return null;

    }


    const number =
        Number(value);


    return Number.isFinite(number)

        ? number

        : null;

}


export function buildProposalCandidateMemory(
    event
) {

    const memory =

        event
            ?.payload
            ?.candidateMemory;


    if(
        !isObject(
            memory
        )
    ){

        return null;

    }


    return {

        id:
            memory.id || null,

        key:

            normalizeText(
                memory.key
            )

            ||

            null,

        status:

            normalizeText(
                memory.status
            )

            ||

            null,

        matchType:

            normalizeText(
                memory.matchType
            )

            ||

            null,

        similarity:
            optionalNumber(
                memory.similarity
            ),

        occurrences:
            optionalNumber(
                memory.occurrences
            )

    };

}
