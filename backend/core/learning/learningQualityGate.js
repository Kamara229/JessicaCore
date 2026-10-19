/*
 * =========================================================
 * JESSICA LEARNING QUALITY GATE FACADE v3
 * =========================================================
 *
 * Proposal
 *   ↓
 * reusable
 *   ↓
 * metrics
 *   ↓
 * dynamic pattern validation
 *   ↓
 * Skill Quality Inspector
 * =========================================================
 */


import {
    validateLearningMetricsQuality
} from "./quality/qualityGateMetrics.js";


import {
    validateDynamicPatternQuality
} from "./quality/dynamicPatternQuality.js";


import {
    inspectLearningSkillQuality
} from "./quality/skillQualityInspector.js";
import {validateGroundedComparisonLearning} from "./quality/groundedComparisonLearningGate.js";


function fail(
    reason,
    extra = {}
) {

    return {

        passed:
            false,

        reason,

        ...extra

    };

}


export function validateLearningQuality(
    proposal
) {

    if(
        !proposal
        ||
        typeof proposal !== "object"
    ){

        return fail(
            "Proposal отсутствует"
        );

    }


    if(
        proposal
            ?.analysis
            ?.reusable !== true
    ){

        return fail(
            "Experience не признан reusable"
        );

    }


    const metricsQuality =

        validateLearningMetricsQuality(
            proposal
        );


    if(
        metricsQuality.passed !== true
    ){

        return metricsQuality;

    }


    const sourceQuality=validateGroundedComparisonLearning(proposal);
    if(!sourceQuality.passed)return fail(sourceQuality.reason,{metrics:metricsQuality.metrics});

    const dynamicQuality =

        validateDynamicPatternQuality(
            proposal
        );


    if(
        dynamicQuality.passed !== true
    ){

        return fail(

            dynamicQuality.reason,

            {
                metrics:
                    metricsQuality.metrics,

                dynamicQuality
            }

        );

    }


    const inspection =

        inspectLearningSkillQuality(
            proposal
        );


    if(
        inspection.passed !== true
    ){

        return fail(

            inspection.reason

            ||

            "Skill не прошёл Quality Inspector",

            {
                metrics:
                    metricsQuality.metrics,

                dynamicQuality,

                inspection
            }

        );

    }


    return {

        passed:
            true,

        reason:
            "Learning Evidence и Skill Quality прошли Quality Gate",

        metrics:
            metricsQuality.metrics,

        dynamicQuality,

        inspection

    };

}
