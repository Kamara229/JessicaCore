import {
    getSupabaseClient
} from "../../storage/supabaseClient.js";



/*
 * =========================================================
 * JESSICA EXPERIENCE HISTORY STORE
 * =========================================================
 *
 * Хранилище версий Experience Skill.
 *
 *
 * Flow:
 *
 * Experience Skill
 *        ↓
 * History Store
 *        ↓
 * Supabase
 *
 *
 * Отвечает:
 *
 * - запись версии;
 * - история;
 * - получение версии;
 * - получение последней версии.
 *
 *
 * НЕ:
 *
 * - ищет Skill;
 * - принимает Learning решения;
 * - вызывает AI.
 *
 * =========================================================
 */



const EXPERIENCE_HISTORY_TABLE =
    "jessica_experience_skill_versions";





/*
 * =========================================================
 * NORMALIZE
 * =========================================================
 */


function normalizeSkillId(
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

        return null;

    }


    return version;

}





function normalizePayload(
    experience
) {


    return {


        ...experience,


        id:
            normalizeSkillId(
                experience.id ||
                experience.skillId
            ),



        version:
            normalizeVersion(
                experience.version
            ),



        previousVersion:
            experience.previousVersion
            ||
            null,



        metadata:
            experience.metadata
            ||
            {},


        mode:
            experience.mode
            ||
            "create"

    };

}







/*
 * =========================================================
 * SAVE VERSION
 * =========================================================
 */


export async function saveExperienceVersion(
    experience
) {


    if (
        !experience ||
        typeof experience !== "object"
    ) {

        throw new Error(
            "Experience отсутствует"
        );

    }



    const payload =
        normalizePayload(
            experience
        );



    if (
        !payload.id
    ) {

        throw new Error(
            "Skill ID отсутствует"
        );

    }



    if (
        !payload.version
    ) {

        throw new Error(
            "Версия Skill отсутствует"
        );

    }





    const supabase =
        getSupabaseClient();





    const {
        data,
        error
    }
    =
    await supabase
        .from(
            EXPERIENCE_HISTORY_TABLE
        )
        .insert({

            skill_id:
                payload.id,


            version:
                payload.version,


            previous_version:
                payload.previousVersion,


            mode:
                payload.mode,


            payload,


            metadata:
                payload.metadata


        })
        .select(
            `
            skill_id,
            version,
            previous_version,
            mode,
            payload,
            metadata,
            created_at
            `
        )
        .single();







    if (
        error
    ) {


        if (
            error.code === "23505"
        ) {


            return {


                success:false,


                reason:
                    "version_exists",


                skillId:
                    payload.id,


                version:
                    payload.version


            };

        }



        throw new Error(

            `History save error: ${error.message}`

        );

    }







    return {


        success:true,


        version:
            data


    };


}







/*
 * =========================================================
 * LOAD HISTORY
 * =========================================================
 */


export async function loadExperienceHistory(
    skillId
) {


    const id =
        normalizeSkillId(
            skillId
        );



    if (
        !id
    ) {

        throw new Error(
            "Skill ID отсутствует"
        );

    }





    const supabase =
        getSupabaseClient();





    const {
        data,
        error
    }
    =
    await supabase
        .from(
            EXPERIENCE_HISTORY_TABLE
        )
        .select(
            `
            skill_id,
            version,
            previous_version,
            mode,
            payload,
            metadata,
            created_at
            `
        )
        .eq(
            "skill_id",
            id
        )
        .order(
            "version",
            {
                ascending:false
            }
        );





    if (
        error
    ) {

        throw new Error(
            `History load error: ${error.message}`
        );

    }





    return Array.isArray(data)
        ? data
        : [];

}







/*
 * =========================================================
 * LOAD VERSION
 * =========================================================
 */


export async function loadExperienceVersion(

    skillId,

    version

) {


    const id =
        normalizeSkillId(
            skillId
        );


    const normalizedVersion =
        normalizeVersion(
            version
        );



    if (
        !id ||
        !normalizedVersion
    ) {

        throw new Error(
            "Некорректный Skill или Version"
        );

    }





    const supabase =
        getSupabaseClient();





    const {
        data,
        error
    }
    =
    await supabase
        .from(
            EXPERIENCE_HISTORY_TABLE
        )
        .select(
            `
            skill_id,
            version,
            previous_version,
            mode,
            payload,
            metadata,
            created_at
            `
        )
        .eq(
            "skill_id",
            id
        )
        .eq(
            "version",
            normalizedVersion
        )
        .maybeSingle();





    if (
        error
    ) {

        throw new Error(
            `Version load error: ${error.message}`
        );

    }





    return data || null;


}







/*
 * =========================================================
 * LOAD LATEST VERSION
 * =========================================================
 */


export async function loadLatestExperienceVersion(
    skillId
) {


    const history =
        await loadExperienceHistory(
            skillId
        );



    if (
        history.length === 0
    ) {

        return null;

    }



    return history[0];

}
