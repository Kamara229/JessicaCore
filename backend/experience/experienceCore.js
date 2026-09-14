import {
    searchExperience
} from "./search/experienceSearch.js";


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
 * Experience Core
 *   ↓
 * Experience Search
 *   ↓
 * подходящий Skill
 *   ↓
 * Experience Result
 *
 *
 * В дальнейшем:
 *
 * Experience Result
 *   ↓
 * Experience Context
 *   ↓
 * PlanningContext
 *   ↓
 * Planner
 *
 *
 * Этот файл НЕ содержит:
 *
 * - алгоритм поиска;
 * - хранение Experience;
 * - AI-анализ обучения;
 * - Planner;
 * - Earnings;
 * - выполнение инструментов.
 *
 *
 * Его задача:
 *
 * объединять Experience-модули
 * в одну рабочую цепочку.
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
            "experience-core"

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
 * task
 *     исходная задача / подзадача
 *
 * experiences
 *     уже загруженный список Skills
 *
 *
 * Experience Core сам не знает,
 * откуда пришли Skills.
 *
 * Это важно:
 *
 * сегодня их может передать JSON,
 * завтра PostgreSQL / Supabase,
 * затем другой Experience Storage.
 *
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
         * Поиск ничего подходящего
         * не обнаружил.
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
                    "experience-search"

            };

        }


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
                "experience-search"

        };


    } catch (error) {


        /*
         * =================================================
         * SEARCH ERROR
         * =================================================
         *
         * Ошибка Experience не должна
         * ломать выполнение Jessica.
         *
         * Если первый слой опыта недоступен,
         * Jessica позже сможет перейти
         * к обычному Planner.
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
