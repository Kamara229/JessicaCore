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
 * Центральный координатор накопленного опыта Jessica.
 *
 *
 * Рабочая цепочка:
 *
 * task
 *   ↓
 * Experience Storage
 *   ↓
 * Active Skills
 *   ↓
 * Experience Search
 *   ↓
 * Matching Skill
 *   ↓
 * Experience Context
 *   ↓
 * Planner
 *
 *
 * Этот модуль НЕ:
 *
 * - работает с Supabase напрямую;
 * - сохраняет Skills;
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


function createEmptyExperienceResult(

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

        planningContext: {

            experience:
                null,

            metadata:
                {}

        }

    };

}


/*
 * =========================================================
 * DEBUG HELPERS
 * =========================================================
 */


function logLoadedExperiences(
    experiences
) {


    console.log(

        "Jessica Experience loaded:",

        JSON.stringify(

            Array.isArray(experiences)

                ? experiences.map(
                    item => ({

                        id:
                            item?.id,

                        name:
                            item?.name,

                        enabled:
                            item?.enabled,

                        keywords:
                            item?.keywords || []

                    })
                )

                : []

            ,

            null,

            2

        )

    );


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
        ).trim();



    if (!cleanTask) {


        console.log(
            "Jessica Experience: empty task"
        );


        return createEmptyExperienceResult(
            "empty-task"
        );


    }



    /*
     * =====================================================
     * LOAD EXPERIENCE
     * =====================================================
     */


    let availableExperiences = [];



    try {


        if (
            Array.isArray(
                experiences
            )
        ) {


            availableExperiences =
                experiences;


            console.log(
                "Jessica Experience: using provided Skills"
            );


        } else {


            availableExperiences =
                await loadExperienceSkills();


            console.log(
                "Jessica Experience: loaded from Storage"
            );


        }


    } catch(error) {


        console.error(

            "Jessica Experience storage error:",

            error

        );


        return createEmptyExperienceResult(
            "storage-error"
        );


    }



    /*
     * =====================================================
     * DEBUG STORAGE RESULT
     * =====================================================
     */


    logLoadedExperiences(
        availableExperiences
    );



    /*
     * =====================================================
     * NO SKILLS
     * =====================================================
     */


    if (

        !Array.isArray(
            availableExperiences
        )

        ||

        availableExperiences.length === 0

    ) {


        console.log(

            "Jessica Experience: no active Skills"

        );


        return createEmptyExperienceResult(
            "no-skills"
        );


    }



    /*
     * =====================================================
     * SEARCH
     * =====================================================
     */


    try {


        console.log(

            "Jessica Experience search task:",

            cleanTask

        );



        const searchResult =
            searchExperience(

                cleanTask,

                availableExperiences

            );



        console.log(

            "Jessica Experience search result:",

            JSON.stringify(

                {

                    found:
                        searchResult?.found,

                    confidence:
                        searchResult?.confidence,

                    skillId:
                        searchResult
                            ?.experience
                            ?.id || null

                },

                null,

                2

            )

        );



        /*
         * =================================================
         * NOT FOUND
         * =================================================
         */


        if (

            !searchResult?.found

            ||

            !searchResult?.experience

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
                    searchResult?.source ||
                    "experience-search",

                reason:
                    "search-no-match",

                planningContext: {

                    experience:
                        null,

                    metadata:
                        {}

                }

            };


        }



        /*
         * =================================================
         * BUILD CONTEXT
         * =================================================
         */


        const planningContext =
            buildExperienceContext(
                searchResult
            );



        /*
         * =================================================
         * SUCCESS
         * =================================================
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


            planningContext


        };



    } catch(error) {


        console.error(

            "Jessica Experience search error:",

            error

        );


        return createEmptyExperienceResult(
            "search-error"
        );


    }


}
