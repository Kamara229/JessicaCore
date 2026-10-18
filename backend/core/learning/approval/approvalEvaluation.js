/*
 * =========================================================
 * JESSICA APPROVAL EVALUATION
 * =========================================================
 *
 * Pure decision layer.
 * No persistence.
 * =========================================================
 */

import {
    reviewLearningProposal
} from "../learningReviewer.js";


import {
    validateLearningQuality
} from "../learningQualityGate.js";


import {
    evaluateLearningAutonomy
} from "../learningAutonomyPolicy.js";


import {
    APPROVAL_ACTION
} from "./approvalConstants.js";


export const APPROVAL_FLOW = {

    REJECT:
        "REJECT",

    KEEP:
        "KEEP",

    REINFORCE:
        "REINFORCE",

    SAVE_VERSION:
        "SAVE_VERSION"

};


export function evaluateApprovalProposal(
    proposal
) {

    const review =

        reviewLearningProposal(
            proposal
        );


    if(
        review.valid !== true
    ){

        return {

            flow:
                APPROVAL_FLOW.REJECT,

            reason:

                review.reason

                ||

                "Proposal не прошёл Reviewer",

            review,

            quality:
                null,

            autonomy:
                null

        };

    }


    const quality =

        validateLearningQuality(
            proposal
        );


    if(
        quality.passed !== true
    ){

        return {

            flow:
                APPROVAL_FLOW.REJECT,

            reason:

                quality.reason

                ||

                "Proposal не прошёл Quality Gate",

            review,

            quality,

            autonomy:
                null

        };

    }


    const autonomy =

        evaluateLearningAutonomy(
            proposal
        );


    if(
        autonomy.action ===
        APPROVAL_ACTION.REINFORCE_ONLY
    ){

        return {

            flow:
                APPROVAL_FLOW.REINFORCE,

            reason:
                autonomy.reason,

            review,

            quality,

            autonomy

        };

    }


    if(
        autonomy.action !==
        APPROVAL_ACTION.AUTO_APPROVE
    ){

        return {

            flow:
                APPROVAL_FLOW.KEEP,

            reason:

                autonomy.reason

                ||

                "Недостаточно Evidence для AUTO_APPROVE",

            review,

            quality,

            autonomy

        };

    }


    return {

        flow:
            APPROVAL_FLOW.SAVE_VERSION,

        reason:
            autonomy.reason,

        review,

        quality,

        autonomy

    };

}
