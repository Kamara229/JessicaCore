import {
    randomUUID
} from "node:crypto";


/*
 * =========================================================
 * JESSICA LEARNING CANDIDATE MEMORY v1
 * =========================================================
 *
 * Persistent accumulation
 * Learning Evidence.
 *
 *
 * Flow:
 *
 * NEW_SKILL Candidate
 *        ↓
 * Candidate Identity
 *        ↓
 * Exact Memory Match
 *        ↓
 * Similarity Match
 *        ↓
 * Merge Evidence
 *        ↓
 * Recalculate Metrics
 *        ↓
 * Persistent Candidate
 *
 *
 * Пока Candidate Memory применяется
 * только к NEW_SKILL.
 *
 *
 * SKILL_IMPROVEMENT имеет отдельную
 * Experience lineage и будет подключён
 * позже после Semantic Improvement layer.
 *
 * =========================================================
 */


import {
    buildCandidateMemoryKey,
    resolveCandidateCategory
} from "./candidateMemory/candidateIdentity.js";


import {
    findBestCandidateMatch
} from "./candidateMemory/candidateMatcher.js";


import {
    mergeLearningCandidates
} from "./candidateMemory/candidateMerge.js";


import {
    findActiveCandidateByKey,
    listActiveCandidatesByCategory,
    createCandidateMemory,
    updateCandidateMemory,
    updateCandidateMemoryStatus
} from "./candidateMemory/candidateStorage.js";


function isObject(
    value
) {

    return (

        value &&
        typeof value === "object" &&
        !Array.isArray(value)

    );

}


/*
 * =========================================================
 * ENRICH QUEUE ITEM
 * =========================================================
 */


function enrichQueueItem({

    queueItem,

    memory,

    candidate,

    matchType,

    similarity = 1

}) {

    const originalEvent =

        queueItem?.event

        ||

        queueItem?.event_json

        ||

        {};


    const event = {

        ...originalEvent,

        confidence:

            candidate.confidence || 0,

        skillId:

            candidate.skillId || null,

        payload: {

            ...(originalEvent.payload || {}),

            skillCandidate:

                candidate,

            candidateMemory: {

                id:

                    memory.id,

                key:

                    memory.candidateKey,

                status:

                    memory.status,

                matchType,

                similarity,

                occurrences:

                    candidate.occurrences || 1

            }

        }

    };


    return {

        ...queueItem,

        skillId:

            candidate.skillId || null,

        confidence:

            candidate.confidence || 0,

        event,

        event_json:

            event

    };

}


/*
 * =========================================================
 * OBSERVE
 * =========================================================
 */


export async function observeLearningCandidate(
    queueItem
) {

    /*
     * Candidate Memory v1:
     *
     * только NEW_SKILL.
     */


    if(
        queueItem?.action !==
        "NEW_SKILL"
    ){

        return {

            success:
                true,

            observed:
                false,

            queueItem

        };

    }


    const event =

        queueItem?.event

        ||

        queueItem?.event_json

        ||

        {};


    const candidate =

        event
            ?.payload
            ?.skillCandidate;


    if(
        !isObject(
            candidate
        )
    ){

        return {

            success:
                false,

            observed:
                false,

            error:
                "NEW_SKILL Candidate отсутствует"

        };

    }


    const candidateKey =

        buildCandidateMemoryKey(
            candidate
        );


    if(
        !candidateKey
    ){

        return {

            success:
                false,

            observed:
                false,

            error:
                "Candidate Memory Key не сформирован"

        };

    }


    const category =

        resolveCandidateCategory(
            candidate
        );


    /*
     * =====================================================
     * 1. EXACT MATCH
     * =====================================================
     */


    const exact =

        await findActiveCandidateByKey(
            candidateKey
        );


    if(
        exact
    ){

        const merged =

            mergeLearningCandidates({

                existing:

                    exact.candidate,

                incoming:

                    candidate,

                exactMatch:
                    true,

                similarity:
                    1

            });


        const updated =

            await updateCandidateMemory({

                id:

                    exact.id,

                candidate:

                    merged,

                skillId:

                    merged.skillId,

                metadata: {

                    ...exact.metadata,

                    lastMatchType:

                        "EXACT"

                }

            });


        return {

            success:
                true,

            observed:
                true,

            created:
                false,

            merged:
                true,

            memory:
                updated,

            candidate:
                merged,

            queueItem:

                enrichQueueItem({

                    queueItem,

                    memory:
                        updated,

                    candidate:
                        merged,

                    matchType:
                        "EXACT",

                    similarity:
                        1

                })

        };

    }


    /*
     * =====================================================
     * 2. SIMILARITY MATCH
     * =====================================================
     */


    const activeCandidates =

        await listActiveCandidatesByCategory(
            category
        );


    const match =

        findBestCandidateMatch({

            candidate,

            candidates:

                activeCandidates

        });


    if(
        match
    ){

        const merged =

            mergeLearningCandidates({

                existing:

                    match.candidate,

                incoming:

                    candidate,

                exactMatch:
                    false,

                similarity:

                    match.similarity

            });


        const updated =

            await updateCandidateMemory({

                id:

                    match.memory.id,

                candidate:

                    merged,

                skillId:

                    merged.skillId,

                metadata: {

                    ...match.memory.metadata,

                    lastMatchType:

                        "SIMILARITY",

                    lastSimilarity:

                        match.similarity

                }

            });


        return {

            success:
                true,

            observed:
                true,

            created:
                false,

            merged:
                true,

            memory:
                updated,

            candidate:
                merged,

            similarity:
                match.similarity,

            queueItem:

                enrichQueueItem({

                    queueItem,

                    memory:
                        updated,

                    candidate:
                        merged,

                    matchType:
                        "SIMILARITY",

                    similarity:
                        match.similarity

                })

        };

    }


    /*
     * =====================================================
     * 3. CREATE NEW MEMORY
     * =====================================================
     */


    const created =

        await createCandidateMemory({

            id:

                randomUUID(),

            candidateKey,

            action:

                "NEW_SKILL",

            skillId:

                candidate.skillId || null,

            category,

            candidate,

            metadata: {

                source:

                    event
                        ?.payload
                        ?.source

                    ||

                    candidate.source

                    ||

                    "learning-pipeline",

                firstQueueItemId:

                    queueItem.id || null,

                firstTraceId:

                    event.traceId || null

            }

        });


    return {

        success:
            true,

        observed:
            true,

        created:
            true,

        merged:
            false,

        memory:
            created,

        candidate,

        queueItem:

            enrichQueueItem({

                queueItem,

                memory:
                    created,

                candidate,

                matchType:
                    "NEW",

                similarity:
                    1

            })

    };

}


/*
 * =========================================================
 * FINAL STATUS
 * =========================================================
 */


export async function setLearningCandidateMemoryStatus({

    id,

    status,

    proposalId = null,

    skillId = null

} = {}) {

    if(
        !id
    ){

        return {

            success:
                false,

            error:
                "Candidate Memory ID отсутствует"

        };

    }


    return updateCandidateMemoryStatus({

        id,

        status,

        metadata: {

            proposalId,

            skillId,

            resolvedAt:

                new Date()
                    .toISOString()

        }

    });

}
