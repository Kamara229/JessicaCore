/*
 * =========================================================
 * JESSICA CANDIDATE MATCHER v1
 * =========================================================
 *
 * Ищет похожий ACTIVE Learning Candidate.
 *
 *
 * Exact Candidate Key
 * сначала проверяется Storage Layer.
 *
 * Этот Matcher используется только
 * как fallback.
 *
 *
 * НЕ:
 *
 * - объединяет Candidates;
 * - работает с Supabase;
 * - вызывает AI.
 *
 * =========================================================
 */


const MATCH_THRESHOLD =
    0.65;


/*
 * =========================================================
 * TOKENIZE
 * =========================================================
 */


function tokenize(
    value
) {

    const text =

        String(
            value || ""
        )
        .toLowerCase();


    return new Set(

        text

            .split(
                /[^a-zа-яё0-9]+/gi
            )

            .map(
                item =>
                    item.trim()
            )

            .filter(
                item =>
                    item.length >= 3
            )

    );

}


function arrayText(
    value
) {

    return Array.isArray(value)

        ? value.join(" ")

        : "";

}


/*
 * =========================================================
 * JACCARD
 * =========================================================
 */


function calculateJaccard(
    leftValue,
    rightValue
) {

    const left =
        tokenize(
            leftValue
        );


    const right =
        tokenize(
            rightValue
        );


    if(
        left.size === 0
        ||
        right.size === 0
    ){

        return 0;

    }


    let intersection = 0;


    for(
        const token
        of left
    ){

        if(
            right.has(
                token
            )
        ){

            intersection += 1;

        }

    }


    const union =

        new Set([
            ...left,
            ...right
        ])
        .size;


    if(
        union === 0
    ){

        return 0;

    }


    return intersection / union;

}


/*
 * =========================================================
 * CANDIDATE SCORE
 * =========================================================
 */


export function calculateCandidateSimilarity(
    left,
    right
) {

    if(
        !left ||
        !right
    ){

        return 0;

    }


    const leftCategory =

        String(
            left.category || ""
        )
        .trim()
        .toLowerCase();


    const rightCategory =

        String(
            right.category || ""
        )
        .trim()
        .toLowerCase();


    const categoryScore =

        leftCategory &&
        rightCategory &&
        leftCategory === rightCategory

            ? 1

            : 0;


    const nameScore =

        calculateJaccard(

            left.name,

            right.name

        );


    const triggerScore =

        calculateJaccard(

            [
                arrayText(
                    left.triggerPatterns
                ),

                arrayText(
                    left.keywords
                )
            ]
            .join(" "),

            [
                arrayText(
                    right.triggerPatterns
                ),

                arrayText(
                    right.keywords
                )
            ]
            .join(" ")

        );


    const workflowScore =

        calculateJaccard(

            arrayText(
                left.workflow
            ),

            arrayText(
                right.workflow
            )

        );


    const score =

        categoryScore * 0.15

        +

        nameScore * 0.25

        +

        triggerScore * 0.35

        +

        workflowScore * 0.25;


    return Number(
        score.toFixed(3)
    );

}


/*
 * =========================================================
 * FIND BEST
 * =========================================================
 */


export function findBestCandidateMatch({

    candidate,

    candidates = []

} = {}) {

    if(
        !candidate ||
        !Array.isArray(
            candidates
        )
    ){

        return null;

    }


    let best = null;


    for(
        const item
        of candidates
    ){

        const storedCandidate =

            item?.candidate

            ||

            item?.candidate_json

            ||

            null;


        if(
            !storedCandidate
        ){

            continue;

        }


        const similarity =

            calculateCandidateSimilarity(

                candidate,

                storedCandidate

            );


        if(
            !best
            ||
            similarity > best.similarity
        ){

            best = {

                memory:
                    item,

                candidate:
                    storedCandidate,

                similarity

            };

        }

    }


    if(
        !best
        ||
        best.similarity <
        MATCH_THRESHOLD
    ){

        return null;

    }


    return best;

}


export function getCandidateMatchThreshold()
{

    return MATCH_THRESHOLD;

}
