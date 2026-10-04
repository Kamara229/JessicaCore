/*
 * =========================================================
 * JESSICA EXPERIENCE STORAGE v5
 * =========================================================
 *
 * Public facade
 * Experience Persistence.
 *
 *
 * Read:
 *
 * loadExperienceSkills
 * getExperienceHistory
 * getExperienceVersion
 * getLatestExperienceVersion
 *
 *
 * Write:
 *
 * saveExperienceSkill
 *
 *
 * Lifecycle:
 *
 * disableExperienceSkill
 *
 * =========================================================
 */


import {
    disableExperience
} from "./supabaseExperienceStore.js";


import {
    readActiveExperienceSkills,
    readExperienceHistory,
    readExperienceVersion,
    readLatestExperienceVersion
} from "./experiencePersistence/experienceReader.js";


import {
    persistExperienceSkill
} from "./experiencePersistence/experienceWriter.js";


import {
    findExperienceByProposalId
} from "./experiencePersistence/experienceIdempotency.js";


import {
    normalizeExperienceText
} from "./experiencePersistence/experienceNormalizer.js";





/*
 * =========================================================
 * ACTIVE SKILLS
 * =========================================================
 */


export async function loadExperienceSkills()
{

    return readActiveExperienceSkills();

}





/*
 * =========================================================
 * SAVE
 * =========================================================
 */


export async function saveExperienceSkill(
    experience
) {

    try {


        return await persistExperienceSkill(
            experience
        );


    }catch(error){


        console.error(

            "Experience save error:",

            error

        );


        return {

            success:
                false,

            existing:
                false,

            error:

                error?.message

                ||

                "Experience Storage error"

        };

    }

}





/*
 * =========================================================
 * IDEMPOTENCY LOOKUP
 * =========================================================
 */


export async function getExperienceByProposalId(
    proposalId
) {

    return findExperienceByProposalId(
        proposalId
    );

}





/*
 * =========================================================
 * DISABLE
 * =========================================================
 */


export async function disableExperienceSkill(
    skillId
) {

    const id =

        normalizeExperienceText(
            skillId
        );


    if(
        !id
    ){

        return {

            success:
                false,

            error:
                "Skill ID отсутствует"

        };

    }


    return disableExperience(
        id
    );

}





/*
 * =========================================================
 * HISTORY
 * =========================================================
 */


export async function getExperienceHistory(
    skillId
) {

    return readExperienceHistory(
        skillId
    );

}


export async function getExperienceVersion(
    skillId,
    version
) {

    return readExperienceVersion(

        skillId,

        version

    );

}


export async function getLatestExperienceVersion(
    skillId
) {

    return readLatestExperienceVersion(
        skillId
    );

}
