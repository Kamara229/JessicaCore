/*
 * =========================================================
 * JESSICA LEARNING PROPOSAL ANALYSIS FACADE v3
 * =========================================================
 */


import {
    isObject,
    normalizeText
} from "./proposalUtils.js";


import {
    buildProposalCurrentMetrics
} from "./analysisMetrics.js";


import {
    buildProposalCandidateMemory
} from "./analysisCandidateMemory.js";


import {
    buildProposalDiscovery
} from "./analysisDiscovery.js";


export function buildProposalAnalysis({

    event,

    candidate,

    confidence,

    action

}) {

    const originalAnalysis =

        isObject(
            event?.analysis
        )

            ? event.analysis

            : {};


    const payloadSource =

        normalizeText(
            event?.payload?.source
        );


    const metrics =

        buildProposalCurrentMetrics({

            originalAnalysis,

            candidate,

            confidence

        });


    const discovery =

        buildProposalDiscovery({

            event,

            candidate

        });


    return {

        reusable:

            event?.reusable === true

            ||

            originalAnalysis?.reusable === true,

        reason:

            normalizeText(
                event?.reason
            )

            ||

            normalizeText(
                originalAnalysis?.reason
            ),

        confidence:
            metrics.confidence,

        action,

        analysisType:

            normalizeText(
                originalAnalysis?.analysisType
            )

            ||

            (
                payloadSource ===
                "ai-pattern-discovery"

                    ? "AI_PATTERN_DISCOVERY"

                    : null
            ),

        improvementType:

            normalizeText(
                originalAnalysis?.improvementType
            )

            ||

            normalizeText(
                candidate?.improvementType
            )

            ||

            null,

        discoveryRequired:

            discovery?.resolved === true

                ? false

                : (
                    originalAnalysis
                        ?.discoveryRequired === true
                ),

        metrics,

        source:

            payloadSource

            ||

            normalizeText(
                originalAnalysis?.source
            )

            ||

            "experience-analyzer",

        discovery,

        evolution:

            isObject(
                candidate?.evolution
            )

                ? {
                    ...candidate.evolution
                }

                : null,

        candidateMemory:

            buildProposalCandidateMemory(
                event
            )

    };

}
