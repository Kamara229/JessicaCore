/*
 * =========================================================
 * JESSICA EXPERIENCE SEARCH v0.4
 * =========================================================
 *
 * Центральный координатор поиска Experience.
 *
 *
 * Flow:
 *
 * Task
 *   ↓
 * Experience Matcher
 *   ↓
 * Compare Skills
 *   ↓
 * Best Match
 *   ↓
 * Confidence Threshold
 *   ↓
 * found / not found
 *
 *
 * Вся специализированная логика вынесена:
 *
 * experienceSearch/
 *
 * ├── experienceText.js
 * │     → текст и semantic concepts
 * │
 * └── experienceMatcher.js
 *       → scoring и confidence
 *
 *
 * Этот файл НЕ:
 *
 * - нормализует текст;
 * - содержит словарь RU / EN;
 * - считает phrase matching;
 * - рассчитывает score самостоятельно;
 * - вызывает AI;
 * - читает Storage;
 * - изменяет Skills.
 *
 * =========================================================
 */


import {
    calculateExperienceMatch
} from "./experienceSearch/experienceMatcher.js";


/*
 * =========================================================
 * CONFIG
 * =========================================================
 */


const MIN_MATCH_CONFIDENCE =
    0.35;


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
 * EMPTY RESULT
 * =========================================================
 */


function createNotFoundResult(
    match = createEmptyMatch()
) {

    return {

        found:
            false,

        experience:
            null,

        confidence:
            match?.confidence || 0,

        matchedTerms:
            Array.isArray(
                match?.matchedTerms
            )
                ? match.matchedTerms
                : [],

        matchedPhrases:
            Array.isArray(
                match?.matchedPhrases
            )
                ? match.matchedPhrases
                : [],

        source:
            "experience-search"

    };

}


/*
 * =========================================================
 * FOUND RESULT
 * =========================================================
 */


function createFoundResult(
    experience,
    match
) {

    return {

        found:
            true,

        experience,

        confidence:
            match?.confidence || 0,

        matchedTerms:
            Array.isArray(
                match?.matchedTerms
            )
                ? match.matchedTerms
                : [],

        matchedPhrases:
            Array.isArray(
                match?.matchedPhrases
            )
                ? match.matchedPhrases
                : [],

        source:
            "experience-search"

    };

}


/*
 * =========================================================
 * VALID EXPERIENCE
 * =========================================================
 */


function isUsableExperience(
    experience
) {

    if (
        !experience ||
        typeof experience !== "object"
    ) {

        return false;

    }


    if (
        experience.enabled === false
    ) {

        return false;

    }


    return true;

}


/*
 * =========================================================
 * FIND BEST EXPERIENCE
 * =========================================================
 */


function findBestExperience(
    task,
    experiences
) {

    let bestExperience =
        null;


    let bestMatch =
        createEmptyMatch();


    for (
        const experience
        of experiences
    ) {

        if (
            !isUsableExperience(
                experience
            )
        ) {

            continue;

        }


        const match =
            calculateExperienceMatch(

                task,

                experience

            );


        if (
            match?.confidence >
            bestMatch.confidence
        ) {

            bestExperience =
                experience;


            bestMatch =
                match;

        }

    }


    return {

        experience:
            bestExperience,

        match:
            bestMatch

    };

}


/*
 * =========================================================
 * LOG BEST MATCH
 * =========================================================
 */


function logBestMatch(
    experience,
    match
) {

    if (
        !experience
    ) {

        return;

    }


    console.log(
        "Jessica Experience best match:",
        JSON.stringify({

            skillId:
                experience?.id || null,

            confidence:
                match?.confidence || 0,

            matchedTerms:
                match?.matchedTerms || [],

            matchedPhrases:
                match?.matchedPhrases || []

        })
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

    /*
     * =====================================================
     * INVALID INPUT
     * =====================================================
     */


    if (
        typeof task !== "string" ||
        !task.trim() ||
        !Array.isArray(
            experiences
        ) ||
        experiences.length === 0
    ) {

        return createNotFoundResult();

    }


    /*
     * =====================================================
     * BEST MATCH
     * =====================================================
     */


    const {
        experience,
        match
    } =
        findBestExperience(

            task,

            experiences

        );


    /*
     * =====================================================
     * DIAGNOSTIC LOG
     * =====================================================
     */


    logBestMatch(

        experience,

        match

    );


    /*
     * =====================================================
     * NO MATCH
     * =====================================================
     */


    if (
        !experience
    ) {

        return createNotFoundResult(
            match
        );

    }


    /*
     * =====================================================
     * BELOW THRESHOLD
     * =====================================================
     */


    if (
        match.confidence <
        MIN_MATCH_CONFIDENCE
    ) {

        return createNotFoundResult(
            match
        );

    }


    /*
     * =====================================================
     * FOUND
     * =====================================================
     */


    return createFoundResult(

        experience,

        match

    );

    }
