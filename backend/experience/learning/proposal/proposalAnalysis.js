/*
 * =========================================================
 * JESSICA LEARNING PROPOSAL ANALYSIS v2
 * =========================================================
 *
 * Формирует диагностический контекст
 * Learning Proposal.
 *
 *
 * Здесь хранится:
 *
 * - источник Candidate;
 * - причина Learning;
 * - актуальные Learning Metrics;
 * - Pattern Discovery;
 * - Candidate Memory;
 * - Improvement Context.
 *
 *
 * ВАЖНО:
 *
 * Experience Analyzer выполняется
 * ДО Candidate Memory.
 *
 * Поэтому после accumulation:
 *
 * candidate.metrics
 *
 * являются более свежими,
 * чем:
 *
 * event.analysis.metrics
 *
 *
 * Актуальные Candidate metrics
 * должны иметь приоритет.
 *
 *
 * НЕ:
 *
 * - создаёт Skill;
 * - принимает Approval;
 * - изменяет Candidate;
 * - работает с Supabase.
 *
 * =========================================================
 */


import {
    isObject,
    normalizeText,
    normalizeUnit,
    normalizePositiveInteger
} from "./proposalUtils.js";





/*
 * =========================================================
 * OPTIONAL NUMBER
 * =========================================================
 */


function optionalNumber(
    value
) {

    if(
        value === undefined
        ||
        value === null
    ){

        return null;

    }


    const number =
        Number(value);


    return Number.isFinite(number)

        ? number

        : null;

}





/*
 * =========================================================
 * CURRENT METRICS
 * =========================================================
 *
 * Base:
 *
 * Analyzer metrics
 *
 *
 * Override:
 *
 * Candidate Memory / Candidate Builder
 *
 *
 * Благодаря этому accumulation
 * сразу виден Reviewer / Quality Gate /
 * Autonomy / Diagnostics.
 *
 * =========================================================
 */


function buildCurrentMetrics({

    originalAnalysis,

    candidate,

    confidence

}) {

    const originalMetrics =

        isObject(
            originalAnalysis?.metrics
        )

            ? originalAnalysis.metrics

            : {};


    const candidateOccurrences =

        optionalNumber(
            candidate?.occurrences
        );


    const candidateSuccessCount =

        optionalNumber(
            candidate?.successCount
        );


    const candidateFailureCount =

        optionalNumber(
            candidate?.failureCount
        );


    const candidateSuccessRate =

        optionalNumber(
            candidate?.successRate
        );


    const candidateMaturity =

        optionalNumber(
            candidate?.maturity
        );


    const candidateMatchScore =

        optionalNumber(
            candidate?.matchScore
        );


    const candidateConfidence =

        optionalNumber(
            candidate?.confidence
        );


    return {


        ...originalMetrics,


        occurrences:

            candidateOccurrences !== null

                ? Math.max(
                    Math.floor(
                        candidateOccurrences
                    ),
                    0
                )

                : normalizePositiveInteger(
                    originalMetrics.occurrences
                ),


        successCount:

            candidateSuccessCount !== null

                ? Math.max(
                    candidateSuccessCount,
                    0
                )

                : Math.max(
                    Number(
                        originalMetrics.successCount || 0
                    ),
                    0
                ),


        failureCount:

            candidateFailureCount !== null

                ? Math.max(
                    candidateFailureCount,
                    0
                )

                : Math.max(
                    Number(
                        originalMetrics.failureCount || 0
                    ),
                    0
                ),


        successRate:

            candidateSuccessRate !== null

                ? normalizeUnit(
                    candidateSuccessRate
                )

                : normalizeUnit(
                    originalMetrics.successRate
                ),


        maturity:

            candidateMaturity !== null

                ? normalizeUnit(
                    candidateMaturity
                )

                : normalizeUnit(
                    originalMetrics.maturity
                ),


        maturityLevel:

            normalizeText(
                candidate?.maturityLevel
            )

            ||

            normalizeText(
                originalMetrics.maturityLevel
            )

            ||

            null,


        matchScore:

            candidateMatchScore !== null

                ? normalizeUnit(
                    candidateMatchScore
                )

                : normalizeUnit(
                    originalMetrics.matchScore
                ),


        confidence:

            candidateConfidence !== null

                ? normalizeUnit(
                    candidateConfidence
                )

                : normalizeUnit(
                    confidence
                )

    };

}





/*
 * =========================================================
 * CANDIDATE MEMORY
 * =========================================================
 */


function buildCandidateMemoryAnalysis(
    event
) {

    const memory =

        event
            ?.payload
            ?.candidateMemory;


    if(
        !isObject(
            memory
        )
    ){

        return null;

    }


    return {

        id:

            memory.id

            ||

            null,


        key:

            normalizeText(
                memory.key
            )

            ||

            null,


        status:

            normalizeText(
                memory.status
            )

            ||

            null,


        matchType:

            normalizeText(
                memory.matchType
            )

            ||

            null,


        similarity:

            optionalNumber(
                memory.similarity
            ),


        occurrences:

            optionalNumber(
                memory.occurrences
            )

    };

}





/*
 * =========================================================
 * BUILD ANALYSIS
 * =========================================================
 */


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


    const candidateMemory =

        buildCandidateMemoryAnalysis(
            event
        );


    const metrics =

        buildCurrentMetrics({

            originalAnalysis,

            candidate,

            confidence

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



        /*
         * Если Dynamic Discovery уже
         * успешно завершён, Proposal
         * больше не "requires discovery".
         */


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



        /*
         * =================================================
         * PATTERN DISCOVERY
         * =================================================
         */


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

                : null,



        /*
         * =================================================
         * CANDIDATE MEMORY
         * =================================================
         */


        candidateMemory

    };

}
