/*
 * =========================================================
 * JESSICA EXPERIENCE IDEMPOTENCY v1
 * =========================================================
 *
 * Proposal ID
 *      ↓
 * immutable Experience History
 *      ↓
 * existing Skill Version
 *
 * =========================================================
 */


import {
    loadExperienceVersionByProposalId
} from "../supabaseExperienceHistoryStore.js";


function normalizeText(
    value
) {

    return String(
        value || ""
    )
    .trim();

}


export async function findExperienceByProposalId(
    proposalId
) {

    const id =

        normalizeText(
            proposalId
        );


    if(
        !id
    ){

        return null;

    }


    return loadExperienceVersionByProposalId(
        id
    );

}
