/*
 * =========================================================
 * JESSICA SKILL TRACE RESULT HELPERS
 * =========================================================
 */


import {
    getLearningSkillId
} from "./skillIdentity.js";


export function safeLearningArray(
    value
) {

    return Array.isArray(value)

        ? value

        : [];

}


export function getResultExperience(
    result
) {

    const value =
        result?.experience;


    return (

        value
        &&
        typeof value === "object"
        &&
        !Array.isArray(value)

    )

        ? value

        : {};

}


export function resultUsesSkill(
    result,
    skillId
) {

    const experience =

        getResultExperience(
            result
        );


    if(
        experience.used !== true
    ){

        return false;

    }


    return safeLearningArray(
        experience.skills
    )
        .some(
            skill =>
                getLearningSkillId(
                    skill
                )
                ===
                skillId
        );

}


export function collectResultTools(
    result
) {

    const tools =

        safeLearningArray(
            result
                ?.executionMeta
                ?.usedTools
        );


    return [

        ...new Set(

            tools
                .map(
                    item =>
                        String(
                            item || ""
                        )
                            .trim()
                )
                .filter(Boolean)

        )

    ];

}
