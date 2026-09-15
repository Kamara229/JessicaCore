/*
 * =========================================================
 * JESSICA EXPERIENCE SEARCH v0.2
 * =========================================================
 *
 * Улучшенный поиск накопленного опыта.
 *
 * Поддерживает:
 *
 * - RU/EN synonyms
 * - weighted matching
 * - phrase matching
 * - Skill metadata priority
 *
 *
 * Не делает:
 *
 * - Storage
 * - Learning
 * - Planner
 *
 * =========================================================
 */


const MIN_MATCH_CONFIDENCE = 0.35;



/*
 * =========================================================
 * SYNONYMS
 * =========================================================
 */


const SYNONYMS = {


    "официальный":
        [
            "official"
        ],


    "сайт":
        [
            "website",
            "site"
        ],


    "найти":
        [
            "find",
            "search"
        ],


    "поиск":
        [
            "search"
        ],


    "проверить":
        [
            "verify",
            "verification"
        ],


    "домен":
        [
            "domain"
        ],


    "ссылка":
        [
            "url",
            "link"
        ]


};



/*
 * =========================================================
 * NORMALIZE
 * =========================================================
 */


function normalizeText(
    value
) {

    return String(
        value || ""
    )
        .toLowerCase()
        .replace(
            /[^a-zа-яё0-9\s_-]/gi,
            " "
        )
        .replace(
            /\s+/g,
            " "
        )
        .trim();

}



/*
 * =========================================================
 * TOKENIZE
 * =========================================================
 */


function tokenize(
    value
) {

    const text =
        normalizeText(
            value
        );


    if (!text) {

        return [];

    }


    return text
        .split(" ")
        .filter(
            token =>
                token.length >= 3
        );

}



/*
 * =========================================================
 * EXPAND TOKEN
 * =========================================================
 */


function expandToken(
    token
) {


    return [

        token,

        ...(SYNONYMS[token] || [])

    ];

}



/*
 * =========================================================
 * BUILD SEARCH PROFILE
 * =========================================================
 */


function buildExperienceProfile(
    experience
) {


    return {


        name:
            normalizeText(
                experience.name
            ),


        description:
            normalizeText(
                experience.description
            ),


        keywords:
            normalizeText(
                (
                    experience.keywords || []
                )
                .join(" ")
            ),


        tags:
            normalizeText(
                (
                    experience.tags || []
                )
                .join(" ")
            )


    };


}



/*
 * =========================================================
 * CALCULATE MATCH
 * =========================================================
 */


function calculateMatch(
    task,
    experience
) {


    const tokens =
        tokenize(
            task
        );


    if (
        tokens.length === 0
    ) {

        return 0;

    }


    const profile =
        buildExperienceProfile(
            experience
        );


    let score = 0;


    let totalWeight =
        tokens.length * 3;



    tokens.forEach(
        token => {


            const variants =
                expandToken(
                    token
                );


            let matchedWeight = 0;



            /*
             * NAME
             */

            if (
                variants.some(
                    v =>
                        profile.name.includes(v)
                )
            ) {

                matchedWeight = 3;

            }


            /*
             * KEYWORDS
             */

            else if (
                variants.some(
                    v =>
                        profile.keywords.includes(v)
                )
            ) {

                matchedWeight = 2;

            }


            /*
             * TAGS / DESCRIPTION
             */

            else if (
                variants.some(
                    v =>
                        profile.tags.includes(v) ||
                        profile.description.includes(v)
                )
            ) {

                matchedWeight = 1;

            }



            score += matchedWeight;


        }
    );



    return Math.min(
        1,
        score / totalWeight
    );


}



/*
 * =========================================================
 * SEARCH EXPERIENCE
 * =========================================================
 */


export function searchExperience(

    task,

    experiences = []

) {


    if (
        !task ||
        !Array.isArray(
            experiences
        )
    ) {


        return {

            found:false,

            experience:null,

            confidence:0,

            source:
                "experience-search"

        };

    }



    let best =
        null;


    let bestScore =
        0;



    for (
        const experience
        of experiences
    ) {


        if (
            experience?.enabled === false
        ) {

            continue;

        }


        const score =
            calculateMatch(
                task,
                experience
            );


        if (
            score >
            bestScore
        ) {

            best =
                experience;


            bestScore =
                score;

        }

    }



    if (
        !best ||
        bestScore <
            MIN_MATCH_CONFIDENCE
    ) {


        return {

            found:false,

            experience:null,

            confidence:
                bestScore,

            source:
                "experience-search"

        };

    }



    return {

        found:true,

        experience:
            best,

        confidence:
            bestScore,

        source:
            "experience-search"

    };


}
