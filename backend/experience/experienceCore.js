import {
    searchExperience
} from "./search/experienceSearch.js";

import {
    buildExperienceContext
} from "./context/experienceContext.js";


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
 * Experience Search
 *   ↓
 * подходящий Skill
 *   ↓
 * Experience Context
 *   ↓
 * готовый контекст для PlanningContext
 *
 *
 * Этот файл НЕ содержит:
 *
 * - алгоритм поиска;
 * - хранение Skills;
 * - обучение;
 * - Planner;
 * - Earnings;
 * - выполнение инструментов.
 *
 *
 * Его задача:
 *
 * объединять Experience-модули
 * в единую рабочую цепочку.
 *
 * =========================================================
 */


/*
 * =========================================================
 * EMPTY RESULT
 * =========================================================
 */


function createEmptyExperienceResult() {

    return {

        found:
            false,

        experience:
            null,

        confidence:
            0,

        source:
            "experience-core",

        planningContext:
            {

                experience:
                    null,

                metadata:
                    {}

            }

    };

}


/*
 * =========================================================
 * RESOLVE EXPERIENCE
 * =========================================================
 */


export async function resolveExperience(

    task,

    experiences = []

) {


    /*
     * =====================================================
     * INPUT
     * =====================================================
     */


    const cleanTask =
        String(
            task || ""
        ).trim();


    if (!cleanTask) {

        return createEmptyExperienceResult();

    }


    if (
        !Array.isArray(
            experiences
        ) ||
        experiences.length === 0
    ) {

        return createEmptyExperienceResult();

    }


    /*
     * =====================================================
     * EXPERIENCE SEARCH
     * =====================================================
     */


    try {


        const searchResult =
            searchExperience(

                cleanTask,

                experiences

            );


        /*
         * =================================================
         * EXPERIENCE NOT FOUND
         * =================================================
         */


        if (
            !searchResult?.found ||
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

                planningContext:
                    {

                        experience:
                            null,

                        metadata:
                            {}

                    }

            };

        }


        /*
         * =================================================
         * BUILD EXPERIENCE CONTEXT
         * =================================================
         */


        const planningContext =
            buildExperienceContext(
                searchResult
            );


        /*
         * =================================================
         * EXPERIENCE FOUND
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

            planningContext

        };


    } catch (error) {


        /*
         * =================================================
         * ERROR
         * =================================================
         *
         * Experience никогда не должен
         * ломать основную работу Jessica.
         *
         * Если первый слой опыта недоступен,
         * Planner сможет работать без него.
         *
         * =================================================
         */


        console.error(
            "Jessica Experience error:",
            error
        );


        return createEmptyExperienceResult();


    }


}
