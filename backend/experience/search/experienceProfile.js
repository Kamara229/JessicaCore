/*
 * =========================================================
 * JESSICA EXPERIENCE PROFILE BUILDER
 * =========================================================
 *
 * Создаёт поисковый профиль Experience Skill.
 *
 *
 * НЕ:
 *
 * - ищет;
 * - считает confidence;
 * - работает с БД.
 *
 * Только готовит данные для Matcher.
 *
 * =========================================================
 */



import {
    normalizeExperienceStringArray
} from "./experienceText.js";





function safeArray(value){

    return Array.isArray(value)
        ? value
        : [];

}





function mergeTextFields(
    experience
){

    return [

        experience.name,

        experience.description,

        experience.category,


        ...safeArray(
            experience.keywords
        ),


        ...safeArray(
            experience.triggerPatterns
        ),


        ...safeArray(
            experience.workflow
        ),


        ...safeArray(
            experience.validationRules
        ),


        ...safeArray(
            experience.successfulPatterns
        ),


        ...safeArray(
            experience.examples
        )

    ]
    .filter(Boolean);

}








export function buildExperienceProfile(
    experience
){

    if(
        !experience ||
        typeof experience !== "object"
    ){

        return {

            keywords:[]

        };

    }





    return {


        ...experience,


        keywords:

            normalizeExperienceStringArray(

                mergeTextFields(
                    experience
                )

            )

    };


}
