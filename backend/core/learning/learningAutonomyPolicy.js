/*
 * =========================================================
 * JESSICA LEARNING AUTONOMY POLICY FACADE v5
 * =========================================================
 */


import {
    getApprovalExperience,
    getApprovalMetrics
} from "./approval/approvalMetrics.js";


import {
    APPROVAL_ACTION
} from "./approval/approvalConstants.js";


import {
    findAutonomyBlocker
} from "./autonomy/autonomyPreconditions.js";


import {
    decideAutonomousLearning
} from "./autonomy/autonomyDecision.js";


function keepCandidate(
    reason,
    metrics = null
) {

    return {

        action:
            APPROVAL_ACTION.KEEP_CANDIDATE,

        reason,

        metrics

    };

}


export function evaluateLearningAutonomy(
    proposal
) {

    if(
        !proposal
        ||
        typeof proposal !== "object"
    ){

        return keepCandidate(
            "Proposal отсутствует"
        );

    }


    const experience =

        getApprovalExperience(
            proposal
        );


    const metrics =

        getApprovalMetrics(
            proposal
        );


    const blocker =

        findAutonomyBlocker({

            proposal,

            experience,

            metrics

        });


    if(blocker){

        return keepCandidate(
            blocker,
            metrics
        );

    }


    return decideAutonomousLearning({

        proposal,

        action:
            proposal.action,

        metrics

    });

}
