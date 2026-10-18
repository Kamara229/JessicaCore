/*
 * =========================================================
 * JESSICA LEARNING APPROVAL RUNNER FACADE v8
 * =========================================================
 *
 * Batch facade:
 * proposals → single processor → summary.
 * =========================================================
 */


import {
    processLearningProposal
} from "./approval/approvalProcessor.js";


import {
    summarizeApprovalResults
} from "./approval/approvalBatchSummary.js";


export async function runLearningApproval(
    proposals = []
) {

    if(
        !Array.isArray(
            proposals
        )
    ){

        return {

            success:
                false,

            processed:
                0,

            learned:
                0,

            reinforced:
                0,

            candidates:
                0,

            rejected:
                0,

            failed:
                0,

            error:
                "Invalid proposals"

        };

    }


    const results = [];


    for(
        const proposal
        of proposals
    ){

        results.push(

            await processLearningProposal(
                proposal
            )

        );

    }


    return summarizeApprovalResults(

        proposals,

        results

    );

}
