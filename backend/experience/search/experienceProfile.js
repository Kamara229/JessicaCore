/*
 * =========================================================
 * JESSICA EXPERIENCE PROFILE BUILDER v0.4
 * =========================================================
 *
 * Создаёт поисковый профиль Experience Skill.
 *
 *
 * Experience Skill
 *      ↓
 * Search Profile
 *      ↓
 * Matcher
 *
 *
 * Ответственность:
 *
 * - подготовка поисковых полей;
 * - нормализация текстовых массивов;
 * - сохранение полного оригинального Skill.
 *
 *
 * ВАЖНО:
 *
 * Profile не должен разрушать
 * структурированные поля Experience.
 *
 * В частности:
 *
 * examples остаётся массивом объектов.
 *
 *
 * НЕ:
 *
 * - выбирает Skill;
 * - считает confidence;
 * - работает с Storage;
 * - изменяет Experience.
 *
 * =========================================================
 */


import {
    normalizeExperienceStringArray
} from "./experienceSearch/experienceText.js";





/*
 * =========================================================
 * SAFE ARRAY
 * =========================================================
 */


function safeArray(
    value
){

    return Array.isArray(
        value
    )

        ? value

        : [];

}





/*
 * =========================================================
 * OBJECT ARRAY
 * =========================================================
 */


function safeObjectArray(
    value
){

    if(
        !Array.isArray(
            value
        )
    ){

        return [];

    }


    return value

        .filter(

            item =>

                item
                &&
                typeof item === "object"
                &&
                !Array.isArray(item)

        )

        .map(

            item => ({
                ...item
            })

        );

}





/*
 * =========================================================
 * NORMALIZE SEARCH FIELD
 * =========================================================
 */


function normalizeField(
    value
){

    return normalizeExperienceStringArray(
        value
    );

}





/*
 * =========================================================
 * BUILD PROFILE
 * =========================================================
 */


export function buildExperienceProfile(
    experience
){

    if(

        !experience

        ||

        typeof experience !== "object"

        ||

        Array.isArray(
            experience
        )

    ){

        return {

            keywords:
                [],

            triggerPatterns:
                [],

            workflow:
                [],

            successfulPatterns:
                [],

            failurePatterns:
                [],

            examples:
                [],

            searchTerms:
                []

        };

    }


    /*
     * =====================================================
     * PROFILE
     * =====================================================
     *
     * Начинаем с полного Experience.
     *
     * Поэтому новые поля Skill автоматически
     * проходят через Search Profile,
     * если отдельно не нормализуются ниже.
     *
     * =====================================================
     */


    return {

        ...experience,


        /*
         * =================================================
         * SEARCHABLE KNOWLEDGE
         * =================================================
         */


        keywords:

            normalizeField(
                experience.keywords
            ),


        triggerPatterns:

            normalizeField(
                experience.triggerPatterns
            ),


        workflow:

            normalizeField(
                experience.workflow
            ),


        successfulPatterns:

            normalizeField(
                experience.successfulPatterns
            ),


        failurePatterns:

            normalizeField(
                experience.failurePatterns
            ),


        /*
         * =================================================
         * STRUCTURED EVIDENCE
         * =================================================
         *
         * КРИТИЧНО:
         *
         * Старый код применял
         * normalizeExperienceStringArray()
         * к массиву объектов examples.
         *
         * В результате получалось:
         *
         * ["[object Object]"]
         *
         * Теперь структура сохраняется.
         *
         * =================================================
         */


        examples:

            safeObjectArray(
                experience.examples
            ),


        /*
         * =================================================
         * ADDITIONAL SEARCH TERMS
         * =================================================
         */


        searchTerms:

            normalizeExperienceStringArray([

                experience.name,

                experience.category,

                ...safeArray(
                    experience.tags
                )

            ])

    };

}
