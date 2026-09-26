/*
 * =========================================================
 * JESSICA EXPERIENCE STORAGE
 * =========================================================
 *
 * Единый интерфейс хранения Experience Skills.
 *
 *
 * Flow:
 *
 * Experience Core
 *        ↓
 * Experience Storage
 *        ↓
 * Supabase Storage Layer
 *
 *
 * Отвечает:
 *
 * - загрузка Skills;
 * - сохранение версии;
 * - история;
 * - получение версии;
 * - отключение Skill.
 *
 *
 * НЕ:
 *
 * - ищет Skill;
 * - обучает;
 * - вызывает AI;
 * - содержит SQL.
 *
 * =========================================================
 */


import {
    loadExperiences,
    disableExperience
} from "./supabaseExperienceStore.js";


import {
    loadExperienceHistory,
    loadExperienceVersion
} from "./supabaseExperienceHistoryStore.js";


import {
    saveExperienceAtomic
} from "./supabaseExperienceWriter.js";





/*
 * =========================================================
 * NORMALIZE
 * =========================================================
 */


function normalizeText(
    value
) {

    return String(
        value || ""
    )
    .trim();

}




function normalizeVersion(
    value
) {

    const version =
        Number(value);


    if (
        !Number.isInteger(version)
        ||
        version < 1
    ) {

        return 1;

    }


    return version;

}





function normalizeExperience(
    experience
) {


    if (
        !experience ||
        typeof experience !== "object"
    ) {

        return null;

    }



    return {


        ...experience,


        id:
            normalizeText(
                experience.id
            ),


        name:
            normalizeText(
                experience.name
            ),


        version:
            normalizeVersion(
                experience.version
            ),


        previousVersion:
            experience.previousVersion
            ?
            normalizeVersion(
                experience.previousVersion
            )
            :
            null,


        enabled:
            experience.enabled !== false,


        confidence:
            Number(
                experience.confidence || 0
            )

    };


}








/*
 * =========================================================
 * LOAD ACTIVE SKILLS
 * =========================================================
 */


export async function loadExperienceSkills()
{


    try {


        const skills =
            await loadExperiences();



        return Array.isArray(
            skills
        )
            ? skills
            : [];



    } catch(error) {


        console.error(

            "Experience Storage load error:",

            error

        );


        return [];

    }


}







/*
 * =========================================================
 * SAVE EXPERIENCE SKILL
 * =========================================================
 */


export async function saveExperienceSkill(
    experience
) {


    const normalized =
        normalizeExperience(
            experience
        );



    if (
        !normalized
    ) {


        return {

            success:false,

            error:
                "Experience Skill отсутствует"

        };

    }



    if (
        !normalized.id
    ) {


        return {

            success:false,

            error:
                "Experience Skill ID отсутствует"

        };

    }



    if (
        !normalized.name
    ) {


        return {

            success:false,

            error:
                "Experience Skill name отсутствует"

        };

    }



    try {


        const result =
            await saveExperienceAtomic(
                normalized
            );



        if (
            !result?.success
        ) {


            return {

                success:false,

                error:
                    result?.error ||
                    "Не удалось сохранить Experience"

            };

        }




        return {


            success:true,


            skillId:
                normalized.id,


            version:
                normalized.version,


            experience:
                normalized,


            storage:
                result



        };



    } catch(error) {


        console.error(

            "Experience save error:",

            error

        );


        return {


            success:false,


            error:
                error.message ||
                "Storage error"


        };


    }


}







/*
 * =========================================================
 * DISABLE SKILL
 * =========================================================
 */


export async function disableExperienceSkill(
    skillId
) {


    if (
        !skillId
    ) {


        return {

            success:false,

            error:
                "Skill ID отсутствует"

        };

    }



    return await disableExperience(
        skillId
    );


}







/*
 * =========================================================
 * LOAD HISTORY
 * =========================================================
 */


export async function getExperienceHistory(
    skillId
) {


    try {


        const history =
            await loadExperienceHistory(
                skillId
            );


        return Array.isArray(
            history
        )
            ? history
            : [];



    } catch(error) {


        console.error(

            "Experience history error:",

            error

        );


        return [];

    }


}







/*
 * =========================================================
 * LOAD VERSION
 * =========================================================
 */


export async function getExperienceVersion(

    skillId,

    version

) {


    try {


        return await loadExperienceVersion(

            skillId,

            normalizeVersion(
                version
            )

        );



    } catch(error) {


        console.error(

            "Experience version error:",

            error

        );


        return null;


    }


}







/*
 * =========================================================
 * GET LATEST VERSION
 * =========================================================
 */


export async function getLatestExperienceVersion(
    skillId
) {


    const history =
        await getExperienceHistory(
            skillId
        );



    if (
        history.length === 0
    ) {

        return null;

    }



    return history
        .sort(

            (a,b) =>

                Number(
                    b.version || 0
                )

                -

                Number(
                    a.version || 0
                )

        )[0];


}
