/*
 * =========================================================
 * JESSICA LEARNING EVENT BUILDER FACADE
 * =========================================================
 */


import {
    normalizeLearningEventPayload,
    resolveLearningEventSkillId,
    resolveLearningEventConfidence
} from "./eventPayload.js";


import {
    compactLearningAnalysis,
    compactLearningDecision
} from "./eventDiagnostics.js";


function normalizeText(
    value
) {

    return String(
        value || ""
    )
        .trim();

}


export function buildLearningEvent({

    trace,

    analysis,

    decision

}) {

    const payload =

        normalizeLearningEventPayload({

            decision,

            analysis

        });


    return {

        id:

            trace?.id

            ||

            null,

        traceId:

            trace?.id

            ||

            trace?.traceId

            ||

            null,

        action:

            decision?.action

            ||

            analysis?.action

            ||

            "IGNORE",

        skillId:

            resolveLearningEventSkillId({

                decision,

                payload

            }),

        confidence:

            resolveLearningEventConfidence({

                decision,

                analysis,

                payload

            }),

        reusable:

            decision?.reusable === true

            ||

            analysis?.reusable === true,

        reason:

            normalizeText(
                decision?.reason
            )

            ||

            normalizeText(
                analysis?.reason
            )

            ||

            "",

        payload,

        analysis:

            compactLearningAnalysis(
                analysis
            ),

        decision:

            compactLearningDecision(
                decision
            ),

        source:
            "learning-trigger",

        createdAt:

            new Date()
                .toISOString()

    };

}
