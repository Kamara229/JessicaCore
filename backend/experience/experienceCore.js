import {
    searchExperience
} from "./search/experienceSearch.js";


import {
    buildExperienceContext
} from "./context/experienceContext.js";


import {
    loadExperienceSkills
} from "./storage/experienceStorage.js";



/*
 * =========================================================
 * JESSICA EXPERIENCE CORE
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
 * Active Skills
 *   ↓
 * Experience Search
 *   ↓
 * Skill Match
 *   ↓
 * Experience Context
 *   ↓
 * Planner
 *
 *
 * Ответственность:
 *
 * - получить активные Skills;
 * - найти подходящий опыт;
 * - построить PlanningContext.
 *
 *
 * НЕ:
 *
 * - сохраняет Experience;
 * - изменяет Skills;
 * - обучает Jessica;
 * - вызывает AI;
 * - выполняет инструменты.
 *
 * =========================================================
 */





/*
 * =========================================================
 * EMPTY RESULT
 * =========================================================
 */


function createEmptyResult(
    reason
) {


    return {


        found:
            false,


        experience:
            null,


        match:
            null,


        confidence:
            0,


        source:
            "experience-core",


        reason:
            reason || "not-found",



        planningContext:
        {

            experience:
                null,


            sourceRules:
                [],


            constraints:
                [],


            instructions:
                [],


            plannerHints:
                [],


            metadata:
            {

                reason:
                    reason || "not-found"

            }

        }


    };


}





/*
 * =========================================================
 * NORMALIZE MATCH
 * =========================================================
 */


function normalizeMatch(
    experience,
    confidence
) {


    if (
        !experience ||
        typeof experience !== "object"
    ) {

        return null;

    }



    return {


        skillId:
            experience.id ||
            null,


        version:
            Number(
                experience.version || 1
            ),


        name:
            experience.name ||
            "",


        confidence:
            Number(
                confidence || 0
            )


    };


}





/*
 * =========================================================
 * LOAD AVAILABLE EXPERIENCE
 * =========================================================
 */


async function getAvailableExperiences(
    experiences
) {


    if (
        Array.isArray(
            experiences
        )
    ) {

        return experiences;

    }



    return await loadExperienceSkills();


}





/*
 * =========================================================
 * RESOLVE EXPERIENCE
 * =========================================================
 */


export async function resolveExperience(

    task,

    experiences = null

) {


    const cleanTask =
        String(
            task || ""
        )
        .trim();





    if (
        !cleanTask
    ) {


        return createEmptyResult(
            "empty-task"
        );


    }






    /*
     * =====================================================
     * STORAGE
     * =====================================================
     */


    let availableExperiences;



    try {


        availableExperiences =
            await getAvailableExperiences(
                experiences
            );



    } catch(error) {



        console.error(

            "Jessica Experience storage error:",

            error

        );



        return createEmptyResult(
            "storage-error"
        );


    }






    if (

        !Array.isArray(
            availableExperiences
        )

        ||

        availableExperiences.length === 0

    ) {



        return createEmptyResult(
            "no-active-skills"
        );


    }







    /*
     * =====================================================
     * SEARCH
     * =====================================================
     */


    let searchResult;



    try {


        searchResult =
            searchExperience(

                cleanTask,

                availableExperiences

            );



    } catch(error) {



        console.error(

            "Jessica Experience search error:",

            error

        );



        return createEmptyResult(
            "search-error"
        );


    }







    if (

        !searchResult?.found

        ||

        !searchResult.experience

    ) {


        return {


            ...createEmptyResult(
                "no-match"
            ),


            confidence:
                Number(
                    searchResult?.confidence || 0
                ),


            source:
                searchResult?.source ||
                "experience-search"



        };


    }








    /*
     * =====================================================
     * CONTEXT
     * =====================================================
     */


    let planningContext;



    try {


        planningContext =
            buildExperienceContext(
                searchResult
            );



    } catch(error) {



        console.error(

            "Jessica Experience context error:",

            error

        );



        planningContext =
        {

            experience:
                null,


            metadata:
            {

                contextError:
                    true

            }

        };


    }








    /*
     * =====================================================
     * SUCCESS
     * =====================================================
     */


    return {


        found:
            true,



        experience:
            searchResult.experience,



        match:

            normalizeMatch(

                searchResult.experience,

                searchResult.confidence

            ),



        confidence:
            Number(
                searchResult.confidence || 0
            ),



        source:
            searchResult.source ||
            "experience-search",



        reason:
            "matched",



        planningContext



    };


}
