/*
 * =========================================================
 * JESSICA EXPERIENCE SEARCH
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
 * Skill Ranking
 *   ↓
 * Confidence Threshold
 *   ↓
 * Experience Match
 *
 *
 * Ответственность:
 *
 * - выбрать лучший Skill;
 * - вернуть confidence;
 * - вернуть причины совпадения.
 *
 *
 * НЕ:
 *
 * - читает Storage;
 * - сохраняет Skills;
 * - изменяет Experience;
 * - вызывает AI.
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



const MAX_RANKING_ITEMS =
    5;






/*
 * =========================================================
 * EMPTY MATCH
 * =========================================================
 */


function emptyMatch() {


    return {


        confidence:
            0,


        matchedTerms:
            [],


        matchedPhrases:
            [],


        reasons:
            []

    };


}






/*
 * =========================================================
 * EMPTY RESULT
 * =========================================================
 */


function notFoundResult(
    match = emptyMatch(),

    ranking = []

) {


    return {


        found:
            false,


        experience:
            null,


        confidence:
            Number(
                match.confidence || 0
            ),


        matchedTerms:
            match.matchedTerms || [],


        matchedPhrases:
            match.matchedPhrases || [],


        matchReasons:
            match.reasons || [],


        ranking,


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


    return Boolean(

        experience &&

        typeof experience === "object" &&

        experience.enabled !== false

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

) {


    return experiences


        .filter(
            isUsableExperience
        )


        .map(

            experience => {


                const match =

                    calculateExperienceMatch(

                        task,

                        experience

                    );



                return {


                    experience,


                    match,


                    confidence:
                        Number(
                            match?.confidence || 0
                        )


                };


            }

        )


        .sort(

            (a,b) =>

                b.confidence -
                a.confidence

        );

}




/*
 * =========================================================
 * BUILD FOUND RESULT
 * =========================================================
 */


function foundResult(
    item,
    ranking
) {


    return {


        found:
            true,


        experience:
            item.experience,


        confidence:
            item.confidence,



        matchedTerms:
            item.match?.matchedTerms || [],



        matchedPhrases:
            item.match?.matchedPhrases || [],



        matchReasons:
            item.match?.reasons || [],



        ranking,



        source:
            "experience-search"



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



    if (

        typeof task !== "string"

        ||

        !task.trim()

        ||

        !Array.isArray(
            experiences
        )

        ||

        experiences.length === 0

    ) {


        return notFoundResult();

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





    const top =
        ranking[0];







    /*
     * =====================================================
     * NO RESULT
     * =====================================================
     */


    if (
        !top
    ) {


        return notFoundResult();

    }






    const publicRanking =

        ranking

            .slice(
                0,
                MAX_RANKING_ITEMS
            )

            .map(

                item => ({

                    skillId:
                        item.experience?.id ||
                        null,


                    confidence:
                        item.confidence,


                    matchedTerms:
                        item.match?.matchedTerms || []

                })

            );







    /*
     * =====================================================
     * BELOW CONFIDENCE
     * =====================================================
     */


    if (

        top.confidence <
        MIN_MATCH_CONFIDENCE

    ) {


        return notFoundResult(

            top.match,

            publicRanking

        );

    }







    /*
     * =====================================================
     * SUCCESS
     * =====================================================
     */


    return foundResult(

        top,

        publicRanking

    );


}
