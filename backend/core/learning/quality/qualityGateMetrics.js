/*
 * =========================================================
 * JESSICA QUALITY GATE METRICS
 * =========================================================
 */

import {
    getApprovalExperience,
    getApprovalMetrics
} from "../approval/approvalMetrics.js";


function isFiniteUnit(
    value
) {

    return (

        Number.isFinite(value)
        &&
        value >= 0
        &&
        value <= 1

    );

}


function failure(
    reason,
    metrics
) {

    return {

        passed:
            false,

        reason,

        metrics

    };

}


export function validateLearningMetricsQuality(
    proposal
) {

    const experience =

        getApprovalExperience(
            proposal
        );


    const metrics =

        getApprovalMetrics(
            proposal
        );


    if(
        !Array.isArray(
            experience.examples
        )
        ||
        experience.examples.length === 0
    ){

        return failure(
            "Learning Evidence отсутствует",
            metrics
        );

    }


    if(
        !isFiniteUnit(
            metrics.confidence
        )
    ){

        return failure(
            "Confidence отсутствует или некорректен",
            metrics
        );

    }


    if(
        !isFiniteUnit(
            metrics.successRate
        )
    ){

        return failure(
            "Success Rate отсутствует или некорректен",
            metrics
        );

    }


    if(
        !isFiniteUnit(
            metrics.maturity
        )
    ){

        return failure(
            "Maturity отсутствует или некорректна",
            metrics
        );

    }


    if(
        !Number.isFinite(
            metrics.occurrences
        )
        ||
        metrics.occurrences < 1
    ){

        return failure(
            "Occurrences отсутствует или некорректен",
            metrics
        );

    }


    return {

        passed:
            true,

        reason:
            "Learning Metrics корректны",

        metrics

    };

}
