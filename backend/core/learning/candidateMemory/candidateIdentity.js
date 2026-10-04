/*
 * =========================================================
 * JESSICA CANDIDATE IDENTITY v1
 * =========================================================
 *
 * Формирует идентичность
 * Learning Candidate.
 *
 *
 * Используется:
 *
 * Candidate Memory
 *      ↓
 * Exact Match
 *      ↓
 * Semantic-like Match fallback
 *
 *
 * НЕ:
 *
 * - работает с БД;
 * - объединяет Candidate;
 * - рассчитывает Confidence;
 * - принимает Learning Decision.
 *
 * =========================================================
 */


function normalizeText(
    value
) {

    return String(
        value || ""
    )
    .trim();

}


function normalizeKeyPart(
    value
) {

    return normalizeText(
        value
    )
    .toLowerCase()
    .replace(
        /[^a-zа-яё0-9]+/gi,
        "-"
    )
    .replace(
        /^-+|-+$/g,
        ""
    )
    .slice(
        0,
        120
    );

}


/*
 * =========================================================
 * CANDIDATE KEY
 * =========================================================
 *
 * Для известных Skills:
 *
 * skill:<skillId>
 *
 *
 * Для Dynamic Candidate,
 * если skillId уже построен:
 *
 * также skill:<skillId>
 *
 *
 * Fallback:
 *
 * category + name
 *
 * =========================================================
 */


export function buildCandidateMemoryKey(
    candidate
) {

    const skillId =

        normalizeKeyPart(
            candidate?.skillId
            ||
            candidate?.id
        );


    if(
        skillId
    ){

        return (

            "skill:"

            +

            skillId

        );

    }


    const category =

        normalizeKeyPart(
            candidate?.category
        )

        ||

        "general";


    const name =

        normalizeKeyPart(
            candidate?.name
        );


    if(
        name
    ){

        return (

            "candidate:"

            +

            category

            +

            ":"

            +

            name

        );

    }


    return null;

}


/*
 * =========================================================
 * CATEGORY
 * =========================================================
 */


export function resolveCandidateCategory(
    candidate
) {

    return (

        normalizeText(
            candidate?.category
        )
        .toLowerCase()

        ||

        "general"

    );

}
