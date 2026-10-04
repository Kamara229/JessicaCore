/*
 * =========================================================
 * JESSICA EXPERIENCE SKILL VALIDATION
 * =========================================================
 *
 * Проверяет контракт Skill
 * непосредственно перед Storage.
 *
 *
 * Это structural validation.
 *
 * Семантическое качество уже должно
 * быть проверено Reviewer / Autonomy.
 *
 * =========================================================
 */


import {
    isObject,
    normalizeText
} from "./skillUtils.js";


export function validateSkillBuildInput(
    proposedExperience
) {


    if(
        !isObject(
            proposedExperience
        )
    ){

        throw new Error(
            "Experience Skill Builder: proposedExperience отсутствует"
        );

    }


    if(
        !normalizeText(
            proposedExperience.name
        )
    ){

        throw new Error(
            "Experience Skill Builder: название Skill отсутствует"
        );

    }


    if(
        !Array.isArray(
            proposedExperience.workflow
        )
        ||
        proposedExperience.workflow.length === 0
    ){

        throw new Error(
            "Experience Skill Builder: workflow отсутствует"
        );

    }


    if(
        !Array.isArray(
            proposedExperience.examples
        )
        ||
        proposedExperience.examples.length === 0
    ){

        throw new Error(
            "Experience Skill Builder: examples отсутствуют"
        );

    }


    return true;

}


export function validateBuiltExperienceSkill(
    skill
) {


    if(
        !isObject(
            skill
        )
    ){

        throw new Error(
            "Experience Skill Builder: Skill не создан"
        );

    }


    if(
        !normalizeText(
            skill.id
        )
    ){

        throw new Error(
            "Experience Skill Builder: Skill ID отсутствует"
        );

    }


    if(
        !normalizeText(
            skill.name
        )
    ){

        throw new Error(
            "Experience Skill Builder: Skill name отсутствует"
        );

    }


    if(
        !Number.isInteger(
            skill.version
        )
        ||
        skill.version < 1
    ){

        throw new Error(
            "Experience Skill Builder: некорректная version"
        );

    }


    if(
        !isObject(
            skill.learning
        )
    ){

        throw new Error(
            "Experience Skill Builder: learning metrics отсутствуют"
        );

    }


    return true;

}
