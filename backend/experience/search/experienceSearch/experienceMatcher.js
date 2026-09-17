/*
 * =========================================================
 * JESSICA EXPERIENCE MATCHER v0.4
 * =========================================================
 *
 * Слой сопоставления задачи с одним Experience Skill.
 *
 *
 * Отвечает за:
 *
 * - phrase matching;
 * - keyword matching;
 * - token overlap;
 * - расчёт confidence;
 * - диагностические matchedTerms;
 * - диагностические matchedPhrases.
 *
 *
 * НЕ отвечает за:
 *
 * - выбор лучшего Skill среди нескольких;
 * - threshold found / not found;
 * - Storage;
 * - AI;
 * - языковую нормализацию.
 *
 *
 * Языковая нормализация находится в:
 *
 * experienceText.js
 *
 * =========================================================
 */


import {
    normalizeExperienceText,
    canonicalizeExperienceTokens,
    uniqueExperienceValues,
    normalizeExperienceStringArray
} from "./experienceText.js";


/*
 * =========================================================
 * SCORE WEIGHTS
 * =========================================================
 */


const SCORE = {

    /*
     * Полное совпадение keyword-фразы.
     *
     * Например:
     *
     * official website
     * ↔
     * официальный сайт
     *
     * current time
     * ↔
     * сейчас час
     */

    KEYWORD_PHRASE:
        0.45,


    /*
     * Однословный keyword.
     */

    KEYWORD_TERM:
        0.25,


    /*
     * Concepts из имени Skill.
     */

    NAME:
        0.25,


    /*
     * Tags.
     */

    TAG:
        0.15,


    /*
     * Description.
     */

    DESCRIPTION:
        0.10

};


/*
 * =========================================================
 * EMPTY MATCH
 * =========================================================
 */


function createEmptyMatch() {

    return {

        confidence:
            0,

        matchedTerms:
            [],

        matchedPhrases:
            []

    };

}


/*
 * =========================================================
 * PHRASE EXISTS
 * =========================================================
 *
 * Проверяет именно последовательность concepts.
 *
 *
 * Пример:
 *
 * task:
 *
 * [search, official, website, blender]
 *
 * phrase:
 *
 * [official, website]
 *
 * → true
 *
 *
 * task:
 *
 * [search, official, website, banana, engine]
 *
 * phrase:
 *
 * [search, engine]
 *
 * → false
 *
 *
 * Поэтому слово Engine в названии продукта
 * больше не создаёт ложный match с:
 *
 * "search engine".
 *
 * =========================================================
 */


function containsExperiencePhrase(
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
 *
 * Сравнивает concepts профиля Skill
 * с concepts пользовательской задачи.
 *
 *
 * Важно:
 *
 * denominator зависит от самого Skill,
 * а НЕ от длины пользовательской задачи.
 *
 * Поэтому название объекта вроде:
 *
 * Quasar Banana Engine ZX-9100
 *
 * не снижает confidence универсального Skill.
 *
 * =========================================================
 */


function calculateTokenOverlap(
    taskTokenSet,
    profileTokens
) {

    const uniqueProfileTokens =
        uniqueExperienceValues(
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
            canonicalizeExperienceTokens(
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
                containsExperiencePhrase(

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
                    normalizeExperienceText(
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
            uniqueExperienceValues(
                matchedTerms
            ),

        matchedPhrases:
            uniqueExperienceValues(
                matchedPhrases
            )

    };

}


/*
 * =========================================================
 * CALCULATE EXPERIENCE MATCH
 * =========================================================
 *
 * Главная публичная функция этого модуля.
 *
 *
 * Получает:
 *
 * task
 * experience
 *
 *
 * Возвращает:
 *
 * {
 *   confidence,
 *   matchedTerms,
 *   matchedPhrases
 * }
 *
 * =========================================================
 */


export function calculateExperienceMatch(
    task,
    experience
) {

    /*
     * =====================================================
     * TASK TOKENS
     * =====================================================
     */


    const taskTokens =
        canonicalizeExperienceTokens(
            task
        );


    if (
        taskTokens.length === 0
    ) {

        return createEmptyMatch();

    }


    const taskTokenSet =
        new Set(
            taskTokens
        );


    /*
     * =====================================================
     * SKILL DATA
     * =====================================================
     */


    const keywords =
        normalizeExperienceStringArray(
            experience?.keywords
        );


    const tags =
        normalizeExperienceStringArray(
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

            canonicalizeExperienceTokens(
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

            canonicalizeExperienceTokens(
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

            canonicalizeExperienceTokens(
                experience?.description
            )

        );


    /*
     * =====================================================
     * FINAL CONFIDENCE
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


    /*
     * =====================================================
     * DIAGNOSTICS
     * =====================================================
     */


    const matchedTerms =
        uniqueExperienceValues([

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
