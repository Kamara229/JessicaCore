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
 * Центральный слой доступа к опыту Jessica.
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
 * Planning Context
 *   ↓
 * Planner
 *
 *
 * НЕ:
 *
 * - хранит Skills;
 * - изменяет Skills;
 * - обучает систему;
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


function emptyExperienceResult(
    reason = "not_found"
) {


    return {


        found:
            false,


        experience:
            null,


        confidence:
            0,


        source:
            "experience-core",


        reason,



        planningContext:
        {

            experience:
                null,


            metadata:
            {

                reason

            }

        }


    };

}





/*
 * =========================================================
 * NORMALIZE EXPERIENCE RESULT
 * =========================================================
 */


function normalizeExperienceResult(
    experience,
    confidence
) {


    return {


        skillId:
            experience?.id ||
            null,


        version:
            Number(
                experience?.version || 0
            ),


        name:
            experience?.name ||
            "",


        confidence:
            Number(
                confidence || 0
            )


    };

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


        return emptyExperienceResult(
            "empty-task"
        );

    }





    /*
     * =====================================================
     * LOAD SKILLS
     * =====================================================
     */


    let availableExperiences;



    try {


        availableExperiences =

            Array.isArray(
                experiences
            )

            ?

            experiences

            :

            await loadExperienceSkills();



    } catch(error) {



        console.error(

            "Jessica Experience Storage error:",

            error

        );



        return emptyExperienceResult(
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


        return emptyExperienceResult(
            "no-skills"
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

            "Jessica Experience Search error:",

            error

        );


        return emptyExperienceResult(
            "search-error"
        );


    }







    if (

        !searchResult?.found

        ||

        !searchResult.experience

    ) {



        return {


            found:
                false,


            experience:
                null,


            confidence:
                Number(
                    searchResult?.confidence || 0
                ),


            source:
                "experience-search",


            reason:
                "no-match",


            planningContext:
            {

                experience:
                    null,


                metadata:
                {

                    searched:
                        true

                }

            }


        };


    }







    /*
     * =====================================================
     * BUILD PLANNER CONTEXT
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

            "Experience Context build error:",

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



        confidence:
            Number(
                searchResult.confidence || 0
            ),



        source:
            searchResult.source ||
            "experience-search",



        reason:
            "matched",



        match:

            normalizeExperienceResult(

                searchResult.experience,

                searchResult.confidence

            ),



        planningContext



    };


}
