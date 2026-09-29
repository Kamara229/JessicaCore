/*
 * =========================================================
 * JESSICA EXPERIENCE SEARCH v0.6
 * =========================================================
 *
 * Ranking layer for Experience Skills.
 *
 * НЕ:
 *
 * - Storage
 * - Learning
 * - AI
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









function emptyMatch()
{

    return {

        confidence:0,

        matchedTerms:[],

        matchedPhrases:[],

        reasons:[]

    };

}









function notFound(
    match = emptyMatch(),
    ranking = []
){

    return {

        found:false,

        experience:null,

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









function isUsableExperience(
    skill
){

    return Boolean(

        skill &&

        typeof skill === "object" &&

        skill.enabled !== false

    );

}









function calculateSkillQuality(
    skill
){

    const confidence =
        Number(
            skill.confidence || 0
        );


    const success =
        Number(
            skill.usage?.successfulRuns || 0
        );


    const failed =
        Number(
            skill.usage?.failedRuns || 0
        );


    const total =
        success + failed;



    let reliability = 0;



    if(
        total > 0
    ){

        reliability =
            success / total;

    }



    return (

        confidence * 0.7

        +

        reliability * 0.3

    );

}









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

            (a,b)=>

                b.rankingScore -
                a.rankingScore

        );

}









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
                    item.experience?.id || null,


                name:
                    item.experience?.name || "",


                confidence:
                    item.confidence,


                rankingScore:
                    item.rankingScore,


                matchedTerms:
                    item.match?.matchedTerms || []

            })

        );

}









export function searchExperience(

    task,

    experiences = []

){

    if(

        typeof task !== "string"

        ||

        !task.trim()

        ||

        !Array.isArray(experiences)

        ||

        experiences.length === 0

    ){

        return notFound();

    }







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






    if(

        top.confidence <

        MIN_MATCH_CONFIDENCE

    ){

        return notFound(

            top.match,

            publicRanking

        );

    }






    const second =
        ranking[1];





    if(

        second &&

        (
            top.rankingScore -
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









    return {


        found:true,


        experience:
            top.experience,


        confidence:
            top.confidence,


        matchedTerms:
            top.match?.matchedTerms || [],


        matchedPhrases:
            top.match?.matchedPhrases || [],


        matchReasons:
            top.match?.reasons || [],


        ranking:

            publicRanking,


        source:
            "experience-search"


    };


}
