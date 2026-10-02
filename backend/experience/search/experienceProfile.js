/*
 * =========================================================
 * JESSICA EXPERIENCE PROFILE BUILDER v0.3
 * =========================================================
 *
 * Создаёт поисковый профиль Experience Skill.
 *
 *
 * Ответственность:
 *
 * - подготовка данных для Matcher;
 * - нормализация полей поиска;
 * - сохранение оригинального Skill.
 *
 *
 * НЕ:
 *
 * - ищет;
 * - считает confidence;
 * - работает с Storage.
 *
 * =========================================================
 */



import {
    normalizeExperienceStringArray
} from "./experienceSearch/experienceText.js";







function safeArray(
    value
){

    return Array.isArray(value)

        ?

        value

        :

        [];

}









function normalizeField(
    value
){

    return normalizeExperienceStringArray(
        value
    );

}









export function buildExperienceProfile(
    experience
){

    if(
        !experience
        ||
        typeof experience !== "object"
    ){

        return {

            keywords:[],

            triggerPatterns:[],

            workflow:[],

            successfulPatterns:[]

        };

    }








    return {


        /*
         * Сохраняем оригинальные данные
         */


        ...experience,








        /*
         * Основные ключи поиска
         */


        keywords:

            normalizeField(

                experience.keywords

            ),








        /*
         * Сценарии запуска
         */


        triggerPatterns:

            normalizeField(

                experience.triggerPatterns

            ),








        /*
         * Этапы процесса
         */


        workflow:

            normalizeField(

                experience.workflow

            ),








        /*
         * Подтверждённые успешные случаи
         */


        successfulPatterns:

            normalizeField(

                experience.successfulPatterns

            ),








        /*
         * Ошибочные сценарии
         */


        failurePatterns:

            normalizeField(

                experience.failurePatterns

            ),








        /*
         * Примеры
         */


        examples:

            normalizeField(

                experience.examples

            ),








        /*
         * Дополнительные поисковые признаки
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
