/*
 * =========================================================
 * JESSICA EXPERIENCE SEARCH v0.7
 * =========================================================
 *
 * Ranking layer for Experience Skills.
 *
 *
 * Ответственность:
 *
 * - построить поисковый профиль;
 * - вычислить Match;
 * - учесть качество Skill;
 * - выбрать лучший Experience Skill;
 * - вернуть ОРИГИНАЛЬНЫЙ полный Skill.
 *
 *
 * ВАЖНО:
 *
 * Search не имеет права превращать
 * Experience Skill в сокращённый DTO.
 *
 * Полный объект Skill должен пройти дальше
 * без потери:
 *
 * - examples;
 * - requiredTools;
 * - learning;
 * - statistics;
 * - metadata;
 * - sourcePriority;
 * - patterns;
 * - будущих полей.
 *
 *
 * НЕ:
 *
 * - Storage;
 * - Learning;
 * - AI;
 * - изменение Experience.
 *
 * =========================================================
 */


import {
    calculateExperienceMatch
} from "./experienceSearch/experienceMatcher.js";


import {
    buildExperienceProfile
} from "./experienceProfile.js";





const MIN_MATCH_CONFIDENCE =
    0.35;


const MIN_CONFIDENCE_GAP =
    0.05;


const MAX_RANKING_ITEMS =
    5;





/*
 * =========================================================
 * EMPTY MATCH
 * =========================================================
 */


function emptyMatch()
{

    return {

        confidence:
            0,

        matchedTerms:
            [],

        matchedPhrases:
            [],

        reasons:
            [],

        details:
            {}

    };

}





/*
 * =========================================================
 * NOT FOUND
 * =========================================================
 */


function notFound(
    match = emptyMatch(),
    ranking = []
){

    return {

        found:
            false,

        experience:
            null,

        confidence:

            Number(
                match?.confidence || 0
            ),

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

        matchReasons:

            Array.isArray(
                match?.reasons
            )

                ? match.reasons

                : [],

        matchDetails:

            match?.details

            &&

            typeof match.details === "object"

                ? match.details

                : {},

        ranking,

        source:
            "experience-search"

    };

}





/*
 * =========================================================
 * USABLE EXPERIENCE
 * =========================================================
 */


function isUsableExperience(
    skill
){

    return Boolean(

        skill

        &&

        typeof skill === "object"

        &&

        !Array.isArray(skill)

        &&

        skill.enabled !== false

    );

}





/*
 * =========================================================
 * RUNTIME STATISTICS
 * =========================================================
 *
 * Канонический контракт:
 *
 * skill.statistics
 *
 *
 * skill.usage временно поддерживаем
 * только для старых сохранённых версий.
 *
 * =========================================================
 */


function getRuntimeStatistics(
    skill
){

    if(
        skill?.statistics
        &&
        typeof skill.statistics === "object"
        &&
        !Array.isArray(
            skill.statistics
        )
    ){

        return skill.statistics;

    }


    if(
        skill?.usage
        &&
        typeof skill.usage === "object"
        &&
        !Array.isArray(
            skill.usage
        )
    ){

        return skill.usage;

    }


    return {};

}





/*
 * =========================================================
 * SKILL QUALITY
 * =========================================================
 */


function calculateSkillQuality(
    skill
){

    const confidence =

        Math.max(

            0,

            Math.min(

                1,

                Number(
                    skill?.confidence || 0
                )

            )

        );


    const statistics =

        getRuntimeStatistics(
            skill
        );


    const success =

        Math.max(

            Number(
                statistics?.successfulRuns || 0
            ),

            0

        );


    const failed =

        Math.max(

            Number(
                statistics?.failedRuns || 0
            ),

            0

        );


    const total =
        success + failed;


    /*
     * Если runtime history ещё отсутствует,
     * не штрафуем новый Skill.
     *
     * Его качество определяется
     * накопленным learning confidence.
     */


    const reliability =

        total > 0

            ? success / total

            : confidence;


    return Math.max(

        0,

        Math.min(

            1,

            (
                confidence * 0.7
                +
                reliability * 0.3
            )

        )

    );

}





/*
 * =========================================================
 * RANK EXPERIENCES
 * =========================================================
 */


function rankExperiences(
    task,
    experiences
){

    return experiences

        .filter(
            isUsableExperience
        )

        .map(

            experience => {


                /*
                 * Profile используется ТОЛЬКО
                 * для Match.
                 *
                 * Оригинальный experience
                 * остаётся неизменным.
                 */


                const profile =

                    buildExperienceProfile(
                        experience
                    );


                const match =

                    calculateExperienceMatch(

                        task,

                        profile

                    );


                const baseConfidence =

                    Number(
                        match?.confidence || 0
                    );


                const quality =

                    calculateSkillQuality(
                        experience
                    );


                const rankingScore =

                    baseConfidence * 0.8

                    +

                    quality * 0.2;


                return {

                    /*
                     * КРИТИЧНО:
                     *
                     * здесь хранится полный
                     * оригинальный Skill.
                     */


                    experience,

                    profile,

                    match,

                    confidence:
                        baseConfidence,

                    rankingScore

                };

            }

        )

        .sort(

            (a, b) =>

                b.rankingScore

                -

                a.rankingScore

        );

}





/*
 * =========================================================
 * PUBLIC RANKING
 * =========================================================
 */


function buildPublicRanking(
    ranking
){

    return ranking

        .slice(
            0,
            MAX_RANKING_ITEMS
        )

        .map(

            item => ({

                skillId:

                    item.experience?.id
                    ||
                    null,

                name:

                    item.experience?.name
                    ||
                    "",

                confidence:

                    item.confidence,

                rankingScore:

                    item.rankingScore,

                matchedTerms:

                    item.match?.matchedTerms
                    ||
                    []

            })

        );

}





/*
 * =========================================================
 * SEARCH
 * =========================================================
 */


export function searchExperience(

    task,

    experiences = []

){

    if(

        typeof task !== "string"

        ||

        !task.trim()

        ||

        !Array.isArray(
            experiences
        )

        ||

        experiences.length === 0

    ){

        return notFound();

    }


    /*
     * =====================================================
     * RANK
     * =====================================================
     */


    const ranking =

        rankExperiences(

            task,

            experiences

        );


    const publicRanking =

        buildPublicRanking(
            ranking
        );


    const top =
        ranking[0];


    if(
        !top
    ){

        return notFound(

            emptyMatch(),

            publicRanking

        );

    }


    /*
     * =====================================================
     * MIN CONFIDENCE
     * =====================================================
     */


    if(

        top.confidence

        <

        MIN_MATCH_CONFIDENCE

    ){

        return notFound(

            top.match,

            publicRanking

        );

    }


    /*
     * =====================================================
     * AMBIGUITY
     * =====================================================
     */


    const second =
        ranking[1];


    if(

        second

        &&

        (
            top.rankingScore
            -
            second.rankingScore
        )

        <

        MIN_CONFIDENCE_GAP

    ){

        return notFound(

            top.match,

            publicRanking

        );

    }


    /*
     * =====================================================
     * SUCCESS
     * =====================================================
     *
     * КРИТИЧНО:
     *
     * Возвращаем top.experience,
     * а НЕ profile.
     *
     * Это полный канонический Skill
     * из Experience Storage.
     *
     * =====================================================
     */


    return {

        found:
            true,

        experience:
            top.experience,

        confidence:
            top.confidence,

        matchedTerms:
            top.match?.matchedTerms
            ||
            [],

        matchedPhrases:
            top.match?.matchedPhrases
            ||
            [],

        matchReasons:
            top.match?.reasons
            ||
            [],

        matchDetails:

            top.match?.details

            &&

            typeof top.match.details === "object"

                ? top.match.details

                : {},

        ranking:
            publicRanking,

        source:
            "experience-search"

    };

}
