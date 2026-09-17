/*
 * =========================================================
 * JESSICA EXPERIENCE SEARCH v0.3
 * =========================================================
 *
 * Детерминированный поиск накопленного Experience.
 *
 *
 * Основные принципы:
 *
 * 1. Длинный пользовательский запрос
 *    не должен автоматически снижать confidence.
 *
 * 2. Название конкретного объекта:
 *
 *    Quasar Banana Engine ZX-9100
 *
 *    не должно мешать найти универсальный Skill:
 *
 *    Verify Official Website
 *
 * 3. Сравнение выполняется по токенам,
 *    а не через String.includes().
 *
 * 4. Многословные выражения проверяются
 *    как фразы.
 *
 *    Поэтому:
 *
 *    "search engine"
 *
 *    НЕ совпадёт только из-за слова "Engine"
 *    в названии продукта.
 *
 * 5. RU / EN варианты приводятся
 *    к общим semantic concepts.
 *
 *
 * Этот модуль НЕ:
 *
 * - вызывает AI;
 * - читает Storage;
 * - обучает Jessica;
 * - строит PlanningContext;
 * - изменяет Skills.
 *
 * =========================================================
 */


const MIN_MATCH_CONFIDENCE =
    0.35;


/*
 * =========================================================
 * SCORE WEIGHTS
 * =========================================================
 */


const SCORE = {

    /*
     * Совпадение полноценной keyword-фразы.
     *
     * Например:
     *
     * official website
     * ↔
     * официальный сайт
     */

    KEYWORD_PHRASE:
        0.45,


    /*
     * Совпадение однословного keyword.
     */

    KEYWORD_TERM:
        0.25,


    /*
     * Совпадение concepts из имени Skill.
     */

    NAME:
        0.25,


    /*
     * Совпадение tags.
     */

    TAG:
        0.15,


    /*
     * Совпадение description.
     */

    DESCRIPTION:
        0.10

};


/*
 * =========================================================
 * NORMALIZE TEXT
 * =========================================================
 */


function normalizeText(
    value
) {

    return String(
        value || ""
    )
        .toLowerCase()

        /*
         * Дефисы и подчёркивания здесь считаем
         * разделителями слов.
         *
         * Для Experience Search нам важнее concepts,
         * чем сохранение идентификатора целиком.
         */

        .replace(
            /[_-]+/g,
            " "
        )

        .replace(
            /[^a-zа-яё0-9\s]/gi,
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
                token.length >= 2
        );

}


/*
 * =========================================================
 * CANONICAL CONCEPT
 * =========================================================
 *
 * Приводит распространённые RU / EN формы
 * к одному смысловому concept.
 *
 *
 * Здесь намеренно нет полного NLP/stemming.
 *
 * Нам нужен:
 *
 * - быстрый;
 * - предсказуемый;
 * - локальный
 *
 * первый слой поиска Experience.
 *
 * =========================================================
 */


function canonicalizeToken(
    token
) {

    const value =
        normalizeText(
            token
        );


    if (!value) {

        return "";

    }


    /*
     * OFFICIAL
     */


    if (
        value === "official" ||
        value === "officially" ||
        value.startsWith(
            "официал"
        )
    ) {

        return "official";

    }


    /*
     * WEBSITE
     */


    if (
        value === "website" ||
        value === "site" ||
        value === "web" ||
        value.startsWith(
            "сайт"
        )
    ) {

        return "website";

    }


    /*
     * SEARCH / FIND
     */


    if (
        value === "search" ||
        value === "find" ||
        value === "finding" ||
        value.startsWith(
            "поиск"
        ) ||
        value.startsWith(
            "ищ"
        ) ||
        value.startsWith(
            "найд"
        ) ||
        value.startsWith(
            "найт"
        )
    ) {

        return "search";

    }


    /*
     * VERIFY
     */


    if (
        value === "verify" ||
        value === "verified" ||
        value === "verification" ||
        value.startsWith(
            "провер"
        )
    ) {

        return "verify";

    }


    /*
     * DOMAIN
     */


    if (
        value === "domain" ||
        value.startsWith(
            "домен"
        )
    ) {

        return "domain";

    }


    /*
     * URL / LINK
     */


    if (
        value === "url" ||
        value === "link" ||
        value.startsWith(
            "ссыл"
        )
    ) {

        return "url";

    }


    /*
     * PROJECT
     */


    if (
        value === "project" ||
        value.startsWith(
            "проект"
        )
    ) {

        return "project";

    }


    return value;

}


/*
 * =========================================================
 * CANONICAL TOKENS
 * =========================================================
 */


function canonicalizeTokens(
    value
) {

    return tokenize(
        value
    )
        .map(
            canonicalizeToken
        )
        .filter(Boolean);

}


/*
 * =========================================================
 * UNIQUE
 * =========================================================
 */


function unique(
    values
) {

    return [
        ...new Set(
            values.filter(Boolean)
        )
    ];

}


/*
 * =========================================================
 * NORMALIZE STRING ARRAY
 * =========================================================
 */


function normalizeStringArray(
    value
) {

    if (
        !Array.isArray(
            value
        )
    ) {

        return [];

    }


    return value
        .map(
            item =>
                String(
                    item || ""
                ).trim()
        )
        .filter(Boolean);

}


/*
 * =========================================================
 * PHRASE EXISTS
 * =========================================================
 *
 * Проверяем последовательность concepts.
 *
 *
 * Пример:
 *
 * task:
 *
 * [search, official, website, project, ...]
 *
 * phrase:
 *
 * [official, website]
 *
 * → true
 *
 *
 * Но:
 *
 * task:
 *
 * [search, official, website, ..., engine]
 *
 * phrase:
 *
 * [search, engine]
 *
 * → false
 *
 *
 * Это устраняет ложный match:
 *
 * Quasar Banana Engine
 * ↔
 * search engine
 *
 * =========================================================
 */


function containsPhrase(
    taskTokens,
    phraseTokens
) {

    if (
        !Array.isArray(
            taskTokens
        ) ||
        !Array.isArray(
            phraseTokens
        ) ||
        phraseTokens.length === 0 ||
        phraseTokens.length >
            taskTokens.length
    ) {

        return false;

    }


    for (
        let start = 0;

        start <=
        taskTokens.length -
        phraseTokens.length;

        start++
    ) {

        let matches =
            true;


        for (
            let offset = 0;

            offset <
            phraseTokens.length;

            offset++
        ) {

            if (
                taskTokens[
                    start + offset
                ] !==
                phraseTokens[offset]
            ) {

                matches =
                    false;

                break;

            }

        }


        if (matches) {

            return true;

        }

    }


    return false;

}


/*
 * =========================================================
 * TOKEN OVERLAP
 * =========================================================
 */


function calculateTokenOverlap(
    taskTokenSet,
    profileTokens
) {

    const uniqueProfileTokens =
        unique(
            profileTokens
        );


    if (
        uniqueProfileTokens.length === 0
    ) {

        return {

            ratio:
                0,

            matched:
                []

        };

    }


    const matched =
        uniqueProfileTokens
            .filter(
                token =>
                    taskTokenSet.has(
                        token
                    )
            );


    return {

        ratio:
            matched.length /
            uniqueProfileTokens.length,

        matched

    };

}


/*
 * =========================================================
 * KEYWORD MATCH
 * =========================================================
 */


function calculateKeywordMatch(
    taskTokens,
    taskTokenSet,
    keywords
) {

    let bestScore =
        0;


    const matchedTerms =
        [];


    const matchedPhrases =
        [];


    for (
        const keyword
        of keywords
    ) {

        const keywordTokens =
            canonicalizeTokens(
                keyword
            );


        if (
            keywordTokens.length === 0
        ) {

            continue;

        }


        /*
         * =================================================
         * MULTI-WORD KEYWORD
         * =================================================
         */


        if (
            keywordTokens.length > 1
        ) {

            if (
                containsPhrase(

                    taskTokens,

                    keywordTokens

                )
            ) {

                bestScore =
                    Math.max(

                        bestScore,

                        SCORE.KEYWORD_PHRASE

                    );


                matchedPhrases.push(
                    normalizeText(
                        keyword
                    )
                );


                matchedTerms.push(
                    ...keywordTokens
                );

            }


            continue;

        }


        /*
         * =================================================
         * SINGLE-WORD KEYWORD
         * =================================================
         */


        const keywordToken =
            keywordTokens[0];


        if (
            taskTokenSet.has(
                keywordToken
            )
        ) {

            bestScore =
                Math.max(

                    bestScore,

                    SCORE.KEYWORD_TERM

                );


            matchedTerms.push(
                keywordToken
            );

        }

    }


    return {

        score:
            bestScore,

        matchedTerms:
            unique(
                matchedTerms
            ),

        matchedPhrases:
            unique(
                matchedPhrases
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

    const taskTokens =
        canonicalizeTokens(
            task
        );


    if (
        taskTokens.length === 0
    ) {

        return {

            confidence:
                0,

            matchedTerms:
                [],

            matchedPhrases:
                []

        };

    }


    const taskTokenSet =
        new Set(
            taskTokens
        );


    const keywords =
        normalizeStringArray(
            experience?.keywords
        );


    const tags =
        normalizeStringArray(
            experience?.tags
        );


    /*
     * =====================================================
     * KEYWORDS
     * =====================================================
     */


    const keywordMatch =
        calculateKeywordMatch(

            taskTokens,

            taskTokenSet,

            keywords

        );


    /*
     * =====================================================
     * NAME
     * =====================================================
     */


    const nameMatch =
        calculateTokenOverlap(

            taskTokenSet,

            canonicalizeTokens(
                experience?.name
            )

        );


    /*
     * =====================================================
     * TAGS
     * =====================================================
     */


    const tagMatch =
        calculateTokenOverlap(

            taskTokenSet,

            canonicalizeTokens(
                tags.join(
                    " "
                )
            )

        );


    /*
     * =====================================================
     * DESCRIPTION
     * =====================================================
     */


    const descriptionMatch =
        calculateTokenOverlap(

            taskTokenSet,

            canonicalizeTokens(
                experience?.description
            )

        );


    /*
     * =====================================================
     * FINAL SCORE
     * =====================================================
     *
     * Важно:
     *
     * знаменатель больше НЕ зависит
     * от количества слов в пользовательской задаче.
     *
     *
     * Поэтому:
     *
     * Quasar Banana Engine ZX-9100
     *
     * не снижает confidence универсального Skill.
     *
     * =====================================================
     */


    let confidence =
        0;


    confidence +=
        keywordMatch.score;


    confidence +=
        nameMatch.ratio *
        SCORE.NAME;


    confidence +=
        tagMatch.ratio *
        SCORE.TAG;


    confidence +=
        descriptionMatch.ratio *
        SCORE.DESCRIPTION;


    confidence =
        Math.min(
            1,
            confidence
        );


    const matchedTerms =
        unique([

            ...keywordMatch.matchedTerms,

            ...nameMatch.matched,

            ...tagMatch.matched,

            ...descriptionMatch.matched

        ]);


    return {

        confidence,

        matchedTerms,

        matchedPhrases:
            keywordMatch.matchedPhrases

    };

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

    /*
     * =====================================================
     * INVALID INPUT
     * =====================================================
     */


    if (
        !task ||
        !Array.isArray(
            experiences
        )
    ) {

        return {

            found:
                false,

            experience:
                null,

            confidence:
                0,

            matchedTerms:
                [],

            matchedPhrases:
                [],

            source:
                "experience-search"

        };

    }


    let best =
        null;


    let bestMatch = {

        confidence:
            0,

        matchedTerms:
            [],

        matchedPhrases:
            []

    };


    /*
     * =====================================================
     * FIND BEST SKILL
     * =====================================================
     */


    for (
        const experience
        of experiences
    ) {

        if (
            experience?.enabled === false
        ) {

            continue;

        }


        const match =
            calculateMatch(

                task,

                experience

            );


        if (
            match.confidence >
            bestMatch.confidence
        ) {

            best =
                experience;


            bestMatch =
                match;

        }

    }


    /*
     * =====================================================
     * DIAGNOSTIC LOG
     * =====================================================
     */


    if (best) {

        console.log(
            "Jessica Experience best match:",
            JSON.stringify({

                skillId:
                    best?.id || null,

                confidence:
                    bestMatch.confidence,

                matchedTerms:
                    bestMatch.matchedTerms,

                matchedPhrases:
                    bestMatch.matchedPhrases

            })
        );

    }


    /*
     * =====================================================
     * BELOW THRESHOLD
     * =====================================================
     */


    if (
        !best ||
        bestMatch.confidence <
        MIN_MATCH_CONFIDENCE
    ) {

        return {

            found:
                false,

            experience:
                null,

            confidence:
                bestMatch.confidence,

            matchedTerms:
                bestMatch.matchedTerms,

            matchedPhrases:
                bestMatch.matchedPhrases,

            source:
                "experience-search"

        };

    }


    /*
     * =====================================================
     * MATCH
     * =====================================================
     */


    return {

        found:
            true,

        experience:
            best,

        confidence:
            bestMatch.confidence,

        matchedTerms:
            bestMatch.matchedTerms,

        matchedPhrases:
            bestMatch.matchedPhrases,

        source:
            "experience-search"

    };

                }
