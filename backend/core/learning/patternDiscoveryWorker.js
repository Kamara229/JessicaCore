/*
 * =========================================================
 * JESSICA PATTERN DISCOVERY WORKER v2
 * =========================================================
 *
 * Фоновая обработка
 * PATTERN_DISCOVERY Queue Item.
 *
 *
 * Flow:
 *
 * PATTERN_DISCOVERY
 *        ↓
 * Discovery Evidence
 *        ↓
 * Experience Pattern Extractor
 *        ↓
 * Dynamic Pattern
 *        ↓
 * Learning Metrics
 *        ↓
 * NEW_SKILL Candidate
 *        ↓
 * Normalized Queue Item
 *
 *
 * После этого обычный Learning Worker
 * может использовать существующий:
 *
 * createLearningProposalFromQueue()
 *
 *
 * НЕ:
 *
 * - сохраняет Proposal;
 * - сохраняет Experience;
 * - принимает AUTO_APPROVE;
 * - работает с версиями.
 *
 * =========================================================
 */


import { hasVerifiedPatternEvidence } from "./patternExtraction/patternEvidencePolicy.js";
import { deriveVerifiedDiscoveryMetrics } from "./confidence/verifiedDiscoveryMetrics.js";

import {
    extractExperiencePattern
} from "./experiencePatternExtractor.js";


import {
    calculateExperienceConfidence,
    calculateExperienceMaturity
} from "./experienceConfidence.js";


import {
    buildNewSkillCandidate
} from "./experienceCandidateBuilder.js";





function isObject(
    value
) {

    return (

        value &&

        typeof value === "object" &&

        !Array.isArray(value)

    );

}





function normalizeNumber(
    value
) {

    const number =
        Number(value);


    return Number.isFinite(number)

        ? number

        : 0;

}





/*
 * =========================================================
 * BUILD SYNTHETIC TRACE
 * =========================================================
 *
 * Candidate Builder нужен только
 * компактный Trace для Example.
 *
 * Полный Execution Trace здесь
 * больше не нужен.
 *
 * =========================================================
 */


function buildTraceFromEvidence({

    evidence,

    traceId

}) {


    return {


        id:

            traceId ||

            null,


        task:

            evidence?.task ||

            "",


        success:

            evidence?.result?.success === true,


        status:

            evidence?.result?.status ||

            null,


        result: {

            success:
                evidence?.result?.success === true,

            verified:
                evidence?.result?.verified === true,

            status:

                evidence?.result?.status ||

                null,


            answer: {


                text:

                    evidence?.answer ||

                    ""

            },


            validation:

                evidence?.validation ||

                null

        },


        executedTools:

            Array.isArray(
                evidence?.tools
            )

                ? [
                    ...evidence.tools
                ]

                : [],


        /*
         * Legacy compatibility for older diagnostics.
         */


        usedTools:

            Array.isArray(
                evidence?.tools
            )

                ? [
                    ...evidence.tools
                ]

                : [],


        steps:

            Array.isArray(
                evidence?.steps
            )

                ? evidence.steps

                : [],


        failures:

            Array.isArray(
                evidence?.failures
            )

                ? evidence.failures

                : []

    };

}





/*
 * =========================================================
 * PROCESS DISCOVERY
 * =========================================================
 */


export async function processPatternDiscovery(
    queueItem,
    {signal=null,extractor=extractExperiencePattern}={}
) {
    if(signal?.aborted){
        return {success:false,resolved:false,cancelled:true,
            reason:"Pattern Discovery cancelled"};
    }


    if(
        !queueItem ||
        typeof queueItem !== "object"
    ){

        return {


            success:
                false,


            resolved:
                false,


            reason:
                "Queue Item отсутствует"

        };

    }



    const event =

        queueItem.event

        ||

        queueItem.event_json

        ||

        {};



    const discovery =

        event
            ?.payload
            ?.discovery;



    if(
        !isObject(
            discovery
        )
    ){

        return {


            success:
                false,


            resolved:
                false,


            reason:
                "Pattern Discovery payload отсутствует"

        };

    }



    const evidence =

        discovery.evidence;



    if(
        !isObject(
            evidence
        )
    ){

        return {


            success:
                false,


            resolved:
                false,


            reason:
                "Pattern Discovery Evidence отсутствует"

        };

    }



    /*
     * The queue may contain legacy "COMPLETED" events whose
     * comparison answer and child validations were lost.
     * Reject them before AI extraction: they cannot prove
     * a reusable successful procedure.
     */
    const trustworthy = hasVerifiedPatternEvidence(evidence);

    if (!trustworthy) {
        return {
            success: true,
            resolved: false,
            ignored: true,
            reason: "Discovery Evidence не содержит проверенного ответа и подтверждения выполнения; обучение запрещено"
        };
    }

    /*
     * =====================================================
     * 1. EXTRACT DYNAMIC PATTERN
     * =====================================================
     */


    const extraction =

        await extractor({

            evidence,
            signal

        });



    if(signal?.aborted || extraction?.stage==="cancelled"){
        return {success:false,resolved:false,cancelled:true,
            reason:"Pattern Discovery cancelled"};
    }

    if(
        !extraction?.success
    ){

        return {


            success:
                false,


            resolved:
                false,


            reason:

                extraction?.reason

                ||

                "Dynamic Pattern extraction failed",


            retryLater:
                extraction?.retryLater === true,


            extraction

        };

    }



    /*
     * Нечего изучать.
     *
     * Это нормальный результат,
     * а не техническая ошибка.
     */


    if(
        extraction.reusable !== true
        ||
        !extraction.pattern
    ){

        return {


            success:
                true,


            resolved:
                false,


            ignored:
                true,


            reason:

                extraction.reason

                ||

                "Reusable Pattern не обнаружен",


            extraction

        };

    }



    /*
     * =====================================================
     * 2. METRICS
     * =====================================================
     */


    // The event may contain stale or unverified aggregate counters.
    // This extraction is supported by exactly one verified trace.
    // Independent additional successes accumulate only via Candidate Memory.
    const verifiedMetrics = deriveVerifiedDiscoveryMetrics(evidence);
    const {occurrences,successRate,confidence,maturity} = verifiedMetrics;

    /*
     * =====================================================
     * 3. BUILD NEW SKILL CANDIDATE
     * =====================================================
     */


    const trace =

        buildTraceFromEvidence({

            evidence,

            traceId:

                discovery.traceId

                ||

                event.traceId

                ||

                null

        });



    const candidate =

        buildNewSkillCandidate({

            pattern:

                extraction.pattern,


            trace,


            confidence,


            maturity,


            occurrences

        });



    if(
        !candidate
    ){

        return {


            success:
                false,


            resolved:
                false,


            reason:
                "Dynamic Skill Candidate не создан",


            extraction

        };

    }



    /*
     * =====================================================
     * 4. NORMALIZE TO NEW_SKILL
     * =====================================================
     *
     * Теперь основной Worker может
     * обрабатывать результат так же,
     * как обычный NEW_SKILL.
     *
     * =====================================================
     */


    const resolvedEvent = {


        ...event,


        action:

            "NEW_SKILL",


        reusable:

            true,


        confidence:

            candidate.confidence || 0,


        reason:

            extraction.reason

            ||

            "Dynamic Experience Pattern обнаружен",


        skillId:

            candidate.skillId

            ||

            null,


        payload: {


            skillCandidate:

                candidate,


            source:

                "ai-pattern-discovery",


            discovery: {


                ...discovery,


                resolved:
                    true,


                pattern:

                    extraction.pattern,


                validation:

                    extraction.validation ||

                    null

            }

        }

    };



    const resolvedQueueItem = {


        ...queueItem,


        action:

            "NEW_SKILL",


        skillId:

            candidate.skillId

            ||

            null,


        confidence:

            candidate.confidence || 0,


        event:

            resolvedEvent,


        /*
         * Supabase rows приходят
         * с event_json.
         *
         * Proposal Builder умеет читать
         * и event, и event_json,
         * но здесь задаём оба поля
         * одинаково для однозначности.
         */


        event_json:

            resolvedEvent

    };



    return {


        success:
            true,


        resolved:
            true,


        ignored:
            false,


        queueItem:

            resolvedQueueItem,


        candidate,


        pattern:

            extraction.pattern,


        extraction

    };

}
