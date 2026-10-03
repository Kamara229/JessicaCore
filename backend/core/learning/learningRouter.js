/*
 * =========================================================
 * JESSICA LEARNING ROUTER v4
 * =========================================================
 *
 * Преобразует Experience Analysis
 * в Learning Event Decision.
 *
 *
 * Actions:
 *
 * NEW_SKILL
 * SKILL_IMPROVEMENT
 * PATTERN_DISCOVERY
 * IGNORE
 *
 *
 * PATTERN_DISCOVERY:
 *
 * Execution Trace
 *      ↓
 * compact Pattern Evidence
 *      ↓
 * Learning Queue
 *      ↓
 * Background Pattern Discovery Worker
 *
 *
 * НЕ:
 *
 * - вызывает AI;
 * - создаёт Skill;
 * - сохраняет Experience;
 * - принимает Approval.
 *
 * =========================================================
 */


import {
    buildPatternEvidence
} from "./patternExtraction/patternEvidence.js";





function normalizeNumber(
    value
) {

    const number =
        Number(value);


    return Number.isFinite(number)

        ? number

        : 0;

}





function normalizeObject(
    value
) {

    if(
        !value ||
        typeof value !== "object" ||
        Array.isArray(value)
    ){

        return {};

    }


    return value;

}





/*
 * =========================================================
 * IGNORE
 * =========================================================
 */


function createIgnoreEvent(
    reason
) {

    return {


        action:
            "IGNORE",


        reusable:
            false,


        confidence:
            0,


        reason:

            reason ||

            "Обучение не требуется",


        payload: {},


        createdAt:

            new Date()
                .toISOString()

    };

}





/*
 * =========================================================
 * BASE EVENT
 * =========================================================
 */


function createBaseEvent(
    analysis
) {

    const candidate =

        normalizeObject(
            analysis?.skillCandidate
        );



    const candidateConfidence =

        normalizeNumber(
            candidate?.confidence
        );



    const analysisConfidence =

        normalizeNumber(
            analysis?.metrics?.confidence
        );



    return {


        confidence:

            candidateConfidence

            ||

            analysisConfidence

            ||

            0,


        reusable:

            analysis?.reusable === true,


        reason:

            analysis?.reason || "",


        payload: {


            skillCandidate:

                candidate,


            source:

                "experience-analyzer"

        },


        createdAt:

            new Date()
                .toISOString()

    };

}





/*
 * =========================================================
 * NEW SKILL
 * =========================================================
 */


function createNewSkillEvent(
    analysis
) {


    return {


        action:
            "NEW_SKILL",


        ...createBaseEvent(
            analysis
        )

    };

}





/*
 * =========================================================
 * SKILL IMPROVEMENT
 * =========================================================
 */


function createSkillImprovementEvent(
    analysis
) {

    const base =

        createBaseEvent(
            analysis
        );



    return {


        action:
            "SKILL_IMPROVEMENT",


        ...base,


        payload: {


            ...base.payload,


            skills:

                analysis
                    ?.skillCandidate
                    ?.skills

                ||

                []

        }

    };

}





/*
 * =========================================================
 * PATTERN DISCOVERY
 * =========================================================
 */


function createPatternDiscoveryEvent(
    trace,
    analysis
) {


    const metrics =

        normalizeObject(
            analysis?.metrics
        );



    const evidence =

        buildPatternEvidence({

            trace,

            metrics

        });



    return {


        action:

            "PATTERN_DISCOVERY",



        reusable:

            false,



        confidence:

            normalizeNumber(
                metrics?.confidence
            ),



        reason:

            analysis?.reason

            ||

            "Требуется самостоятельное извлечение Experience Pattern",



        payload: {


            source:

                "pattern-discovery",



            discovery: {


                evidence,


                metrics,


                traceId:

                    trace?.id ||

                    null

            }

        },



        createdAt:

            new Date()
                .toISOString()

    };

}





/*
 * =========================================================
 * ROUTER
 * =========================================================
 */


export function routeLearningEvent({

    trace = null,

    analysis = null

} = {}) {


    if(
        !analysis ||
        typeof analysis !== "object"
    ){

        return createIgnoreEvent(
            "Нет результата анализа"
        );

    }



    switch(
        analysis.action
    ){


        case "NEW_SKILL":

            return createNewSkillEvent(
                analysis
            );



        case "SKILL_IMPROVEMENT":

            return createSkillImprovementEvent(
                analysis
            );



        case "PATTERN_DISCOVERY":

            return createPatternDiscoveryEvent(
                trace,
                analysis
            );



        case "IGNORE":

            return createIgnoreEvent(
                analysis.reason
            );



        default:

            return createIgnoreEvent(
                "Неизвестный тип Learning Action"
            );

    }

        }
