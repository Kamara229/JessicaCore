/*
 * =========================================================
 * JESSICA LEARNING PROPOSAL ANALYSIS
 * =========================================================
 *
 * Формирует диагностический контекст
 * Learning Proposal.
 *
 *
 * Здесь хранится:
 *
 * - откуда пришёл Candidate;
 * - почему Experience признан reusable;
 * - какие были Learning Metrics;
 * - был ли Dynamic Pattern Discovery;
 * - какой Improvement выполняется.
 *
 *
 * НЕ:
 *
 * - создаёт Skill;
 * - принимает Approval;
 * - меняет Candidate.
 *
 * =========================================================
 */


import {
    isObject,
    normalizeText,
    normalizeUnit,
    normalizePositiveInteger
} from "./proposalUtils.js";


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


    const discovery =

        isObject(
            event?.payload?.discovery
        )

            ? event.payload.discovery

            : null;


    const fallbackMetrics = {


        occurrences:

            normalizePositiveInteger(
                candidate?.occurrences
            ),


        successRate:

            normalizeUnit(
                candidate?.successRate
            ),


        maturity:

            normalizeUnit(
                candidate?.maturity
            ),


        confidence:

            normalizeUnit(
                confidence
            )

    };


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

            normalizeUnit(
                confidence
            ),



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

            originalAnalysis
                ?.discoveryRequired === true,



        metrics:

            isObject(
                originalAnalysis?.metrics
            )

                ? originalAnalysis.metrics

                : fallbackMetrics,



        source:

            payloadSource

            ||

            normalizeText(
                originalAnalysis?.source
            )

            ||

            "experience-analyzer",



        discovery:

            discovery

                ? {


                    resolved:

                        discovery.resolved === true,



                    traceId:

                        discovery.traceId

                        ||

                        event?.traceId

                        ||

                        null,



                    patternId:

                        discovery
                            ?.pattern
                            ?.id

                        ||

                        candidate?.skillId

                        ||

                        null,



                    validation:

                        isObject(
                            discovery.validation
                        )

                            ? discovery.validation

                            : null

                }

                : null

    };

}
