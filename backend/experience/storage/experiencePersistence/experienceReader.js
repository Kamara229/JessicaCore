/*
 * =========================================================
 * JESSICA EXPERIENCE READER v1
 * =========================================================
 *
 * Read-side Experience Persistence.
 *
 * =========================================================
 */


import {
    loadExperiences,
    loadExperienceSkill
} from "../supabaseExperienceStore.js";


import {
    loadExperienceHistory,
    loadExperienceVersion,
    loadLatestExperienceVersion
} from "../supabaseExperienceHistoryStore.js";


import {
    normalizeExperienceText,
    normalizeExperienceVersion
} from "./experienceNormalizer.js";





export async function readActiveExperienceSkills()
{

    const skills =

        await loadExperiences();


    return Array.isArray(skills)

        ? skills

        : [];

}


export async function readActiveExperienceSkill(
    skillId
) {

    const id =

        normalizeExperienceText(
            skillId
        );


    if(
        !id
    ){

        return null;

    }


    return loadExperienceSkill(
        id
    );

}


export async function readExperienceHistory(
    skillId
) {

    const id =

        normalizeExperienceText(
            skillId
        );


    if(
        !id
    ){

        return [];

    }


    /*
     * ВАЖНО:
     *
     * ошибки больше здесь НЕ проглатываются.
     *
     * [] означает только реальное
     * отсутствие History.
     */


    const history =

        await loadExperienceHistory(
            id
        );


    return Array.isArray(history)

        ? history

        : [];

}


export async function readExperienceVersion(
    skillId,
    version
) {

    const id =

        normalizeExperienceText(
            skillId
        );


    const normalizedVersion =

        normalizeExperienceVersion(
            version
        );


    if(
        !id ||
        !normalizedVersion
    ){

        return null;

    }


    return loadExperienceVersion(

        id,

        normalizedVersion

    );

}


export async function readLatestExperienceVersion(
    skillId
) {

    const id =

        normalizeExperienceText(
            skillId
        );


    if(
        !id
    ){

        return null;

    }


    return loadLatestExperienceVersion(
        id
    );

}
