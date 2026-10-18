/*
 * =========================================================
 * JESSICA AUTONOMY PRECONDITIONS
 * =========================================================
 */

import {
    AUTONOMY_THRESHOLDS
} from "./autonomyThresholds.js";


export function findAutonomyBlocker({

    proposal,

    experience,

    metrics

}) {

    if(
        proposal
            ?.analysis
            ?.reusable !== true
    ){

        return "Experience не признан reusable";

    }


    if(
        !experience
            ?.workflow
            ?.length
    ){

        return "Workflow отсутствует";

    }


    if(
        metrics.examplesCount <
        AUTONOMY_THRESHOLDS.minExamples
    ){

        return "Недостаточно подтверждённых Examples";

    }


    if(
        !Number.isFinite(
            metrics.occurrences
        )
        ||
        metrics.occurrences <
        AUTONOMY_THRESHOLDS.minOccurrences
    ){

        return "Недостаточно наблюдений";

    }


    if(
        !Number.isFinite(
            metrics.successRate
        )
    ){

        return "Success Rate отсутствует";

    }


    if(
        metrics.successRate <
        AUTONOMY_THRESHOLDS.minSuccessRate
    ){

        return "Недостаточный показатель успешности";

    }


    if(
        !Number.isFinite(
            metrics.maturity
        )
    ){

        return "Maturity отсутствует";

    }


    if(
        metrics.maturity <
        AUTONOMY_THRESHOLDS.minMaturity
    ){

        return "Недостаточная зрелость Experience";

    }


    if(
        !Number.isFinite(
            metrics.confidence
        )
    ){

        return "Confidence отсутствует";

    }


    return null;

}
