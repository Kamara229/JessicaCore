/*
 * =========================================================
 * JESSICA EXPERIENCE CORE v2
 * =========================================================
 *
 * Центральный координатор Experience слоя.
 *
 *
 * Flow:
 *
 * Task
 *   ↓
 * Experience Storage
 *   ↓
 * Experience Search
 *   ↓
 * Experience Match
 *   ↓
 * Experience Context
 *   ↓
 * Planner
 *
 *
 * НЕ:
 *
 * - сохраняет Experience;
 * - обучает;
 * - вызывает AI;
 * - выполняет инструменты.
 *
 * =========================================================
 */



import {
    searchExperience
} from "./search/experienceSearch.js";


import {
    buildExperienceContext
} from "./context/experienceContext.js";


import {
    loadExperienceSkills
} from "./storage/experienceStorage.js";









function createEmptyResult(
    reason
){

    return {


        found:false,


        experience:null,


        match:null,


        confidence:0,


        source:
            "experience-core",


        reason:
            reason || "not-found",



        planningContext:{

            experience:null,

            sourceRules:[],

            constraints:[],

            instructions:[],

            plannerHints:[],

            metadata:{

                experienceFound:false,

                reason:
                    reason || "not-found"

            }

        }


    };

}









function normalizeMatch(
    result
){

    if(
        !result?.experience
    ){

        return null;

    }



    const experience =
        result.experience;



    return {


        skillId:

            experience.id ||

            null,



        version:

            Number(
                experience.version || 1
            ),



        name:

            experience.name || "",



        confidence:

            Number(
                result.confidence || 0
            ),



        rankingScore:

            Number(
                result.rankingScore || 0
            ),



        matchedTerms:

            result.matchedTerms || [],



        matchedPhrases:

            result.matchedPhrases || [],



        matchReasons:

            result.matchReasons || [],



        matchDetails:

            result.matchDetails || {}

    };

}









async function getAvailableExperiences(
    experiences
){

    if(
        Array.isArray(
            experiences
        )
    ){

        return experiences;

    }



    return await loadExperienceSkills();

}









export async function resolveExperience(

    task,

    experiences = null

){

    const cleanTask =

        String(
            task || ""
        )
        .trim();





    if(
        !cleanTask
    ){

        return createEmptyResult(
            "empty-task"
        );

    }








    let availableExperiences;



    try{


        availableExperiences =

            await getAvailableExperiences(
                experiences
            );



    }catch(error){


        console.error(

            "Jessica Experience storage error:",

            error

        );


        return createEmptyResult(
            "storage-error"
        );

    }








    if(

        !Array.isArray(
            availableExperiences
        )

        ||

        availableExperiences.length === 0

    ){

        return createEmptyResult(
            "no-active-skills"
        );

    }









    let searchResult;



    try{


        searchResult =

            searchExperience(

                cleanTask,

                availableExperiences

            );



    }catch(error){


        console.error(

            "Jessica Experience search error:",

            error

        );


        return createEmptyResult(
            "search-error"
        );


    }








    if(

        !searchResult?.found

        ||

        !searchResult.experience

    ){


        return {


            ...createEmptyResult(
                "no-match"
            ),



            confidence:

                Number(
                    searchResult?.confidence || 0
                ),



            source:

                searchResult?.source
                ||
                "experience-search",



            ranking:

                searchResult?.ranking || []



        };

    }









    let planningContext;



    try{


        planningContext =

            buildExperienceContext(
                searchResult
            );



    }catch(error){


        console.error(

            "Jessica Experience context error:",

            error

        );



        planningContext = {

            experience:null,

            sourceRules:[],

            constraints:[],

            instructions:[],

            plannerHints:[],

            metadata:{

                experienceFound:false,

                contextError:true

            }

        };


    }









    return {


        found:true,


        experience:

            searchResult.experience,



        match:

            normalizeMatch(
                searchResult
            ),



        confidence:

            Number(
                searchResult.confidence || 0
            ),



        source:

            searchResult.source
            ||
            "experience-search",



        reason:

            "matched",



        planningContext



    };


}
