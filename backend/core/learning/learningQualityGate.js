/*
 * =========================================================
 * JESSICA LEARNING QUALITY GATE v2
 * =========================================================
 *
 * Проверяет пригодность Learning Evidence.
 *
 *
 * Reviewer:
 * структурная корректность
 *
 * Quality Gate:
 * целостность evidence и metrics
 *
 * Autonomy Policy:
 * достаточно ли evidence для обучения
 *
 *
 * НЕ:
 *
 * - принимает AUTO_APPROVE;
 * - сохраняет Skill;
 * - работает с Supabase.
 *
 * =========================================================
 */


import {
    getApprovalExperience,
    getApprovalMetrics
} from "./approval/approvalMetrics.js";


function isFiniteUnit(
    value
) {

    return (

        Number.isFinite(value) &&
        value >= 0 &&
        value <= 1

    );

}


export function validateLearningQuality(
    proposal
) {


    if(
        !proposal ||
        typeof proposal !== "object"
    ){

        return {

            passed:
                false,

            reason:
                "Proposal отсутствует"

        };

    }


    if(
        proposal?.analysis?.reusable !== true
    ){

        return {

            passed:
                false,

            reason:
                "Experience не признан reusable"

        };

    }


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

        return {

            passed:
                false,

            reason:
                "Learning Evidence отсутствует"

        };

    }


    const successfulExamples =

        experience.examples.filter(

            item =>
                item?.success === true

        )
        .length;


    if(
        successfulExamples < 1
    ){

        return {

            passed:
                false,

            reason:
                "Нет подтверждённого успешного Example"

        };

    }


    if(
        !isFiniteUnit(
            metrics.confidence
        )
    ){

        return {

            passed:
                false,

            reason:
                "Confidence отсутствует или некорректен"

        };

    }


    if(
        !isFiniteUnit(
            metrics.successRate
        )
    ){

        return {

            passed:
                false,

            reason:
                "Success Rate отсутствует или некорректен"

        };

    }


    if(
        !isFiniteUnit(
            metrics.maturity
        )
    ){

        return {

            passed:
                false,

            reason:
                "Maturity отсутствует или некорректна"

        };

    }


    if(
        !Number.isFinite(
            metrics.occurrences
        )
        ||
        metrics.occurrences < 1
    ){

        return {

            passed:
                false,

            reason:
                "Occurrences отсутствует или некорректен"

        };

    }


    /*
     * =====================================================
     * DYNAMIC PATTERN
     * =====================================================
     *
     * AI-created Pattern обязан
     * предварительно пройти
     * Pattern Validator.
     *
     * =====================================================
     */


    if(
        proposal
            ?.provenance
            ?.dynamicPattern === true
    ){

        const discovery =

            proposal
                ?.analysis
                ?.discovery;


        if(
            discovery?.resolved !== true
        ){

            return {

                passed:
                    false,

                reason:
                    "Dynamic Pattern Discovery не завершён"

            };

        }


        const validation =

            discovery?.validation;


        if(
            !validation ||
            validation.valid !== true
        ){

            return {

                passed:
                    false,

                reason:
                    "Dynamic Pattern не прошёл Validation"

            };

        }

    }


    return {

        passed:
            true,

        reason:
            "Learning Evidence прошёл Quality Gate",

        metrics

    };

}
