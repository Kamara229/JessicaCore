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
 * активные Skills
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
 * - реализацию Supabase;
 * - SQL;
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
 * RESOLVE EXPERIENCE
 * =========================================================
 *
 * Главная точка входа первого слоя
 * собственного опыта Jessica.
 *
 *
 * В обычной работе:
 *
 * resolveExperience(task)
 *
 * Skills автоматически загружаются
 * через Experience Storage.
 *
 *
 * Для тестов можно передать Skills вручную:
 *
 * resolveExperience(
 *     task,
 *     experiences
 * )
 *
 * =========================================================
 */


export async function resolveExperience(

    task,

    experiences = null

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


    /*
     * =====================================================
     * LOAD EXPERIENCE
     * =====================================================
     *
     * Если массив Skills передан вручную,
     * используем его.
     *
     * Иначе загружаем активные Skills
     * из Experience Storage.
     *
     * =====================================================
     */


    let availableExperiences;


    try {


        if (
            Array.isArray(
                experiences
            )
        ) {


            availableExperiences =
                experiences;


        } else {


            availableExperiences =
                await loadExperienceSkills();


        }


    } catch (error) {


        /*
         * Ошибка Storage не должна
         * останавливать Jessica.
         *
         * Первый слой опыта просто
         * считается недоступным.
         */


        console.error(
            "Jessica Experience storage error:",
            error
        );


        return createEmptyExperienceResult();


    }


    /*
     * =====================================================
     * NO EXPERIENCE
     * =====================================================
     */


    if (
        !Array.isArray(
            availableExperiences
        ) ||
        availableExperiences.length === 0
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

                availableExperiences

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
         * SEARCH ERROR
         * =================================================
         *
         * Experience никогда не должен
         * ломать основную работу Jessica.
         *
         * Если первый слой опыта недоступен,
         * Planner позже сможет работать
         * обычным способом.
         *
         * =================================================
         */


        console.error(
            "Jessica Experience search error:",
            error
        );


        return createEmptyExperienceResult();


    }


}
