import {
    getSupabaseClient
} from "../../storage/supabaseClient.js";


/*
 * =========================================================
 * JESSICA EXPERIENCE HISTORY STORE v3
 * =========================================================
 *
 * История версий Experience Skill.
 *
 *
 * Отвечает:
 *
 * - сохранение версии;
 * - получение истории;
 * - получение конкретной версии;
 * - получение последней версии.
 *
 *
 * НЕ:
 *
 * - обучает;
 * - создаёт Skill;
 * - принимает решения.
 *
 * =========================================================
 */


const EXPERIENCE_HISTORY_TABLE =
    "jessica_experience_skill_versions";







function normalizeSkillId(
    value
){

    return String(
        value || ""
    )
    .trim();

}







function normalizeVersion(
    value
){

    const version =
        Number(value);



    if(
        !Number.isInteger(version)
        ||
        version < 1
    ){

        return null;

    }


    return version;

}







function normalizePayload(
    experience
){

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



        mode:

            experience.mode
            ||
            "create",



        metadata:

        {

            ...(experience.metadata || {}),


            historyVersion:

                "v3"

        }


    };

}








function normalizeHistoryItem(
    row
){

    return {


        skillId:

            row.skill_id,


        version:

            Number(
                row.version
            ),



        previousVersion:

            row.previous_version
            ||
            null,



        mode:

            row.mode
            ||
            "create",



        payload:

            row.payload || {},



        metadata:

            row.metadata || {},



        createdAt:

            row.created_at || null


    };

}








/*
 * =========================================================
 * SAVE VERSION
 * =========================================================
 */


export async function saveExperienceVersion(
    experience
){

    if(
        !experience ||
        typeof experience !== "object"
    ){

        throw new Error(
            "Experience отсутствует"
        );

    }





    const payload =

        normalizePayload(
            experience
        );





    if(
        !payload.id
    ){

        throw new Error(
            "Skill ID отсутствует"
        );

    }






    if(
        !payload.version
    ){

        throw new Error(
            "Версия отсутствует"
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

        .select()

        .single();









    if(
        error
    ){

        if(
            error.code === "23505"
        ){

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

            normalizeHistoryItem(
                data
            )


    };


}








/*
 * =========================================================
 * LOAD HISTORY
 * =========================================================
 */


export async function loadExperienceHistory(
    skillId
){

    const id =

        normalizeSkillId(
            skillId
        );





    if(
        !id
    ){

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

        .select("*")

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







    if(
        error
    ){

        throw new Error(

            `History load error: ${error.message}`

        );

    }







    return Array.isArray(data)

        ?

        data.map(
            normalizeHistoryItem
        )

        :

        [];

}








/*
 * =========================================================
 * LOAD VERSION
 * =========================================================
 */


export async function loadExperienceVersion(

    skillId,

    version

){

    const id =

        normalizeSkillId(
            skillId
        );



    const v =

        normalizeVersion(
            version
        );





    if(
        !id ||
        !v
    ){

        throw new Error(
            "Некорректная версия"
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

        .select("*")

        .eq(
            "skill_id",
            id
        )

        .eq(
            "version",
            v
        )

        .maybeSingle();







    if(
        error
    ){

        throw new Error(

            `Version load error: ${error.message}`

        );

    }








    return data

        ?

        normalizeHistoryItem(
            data
        )

        :

        null;


}








/*
 * =========================================================
 * LOAD LATEST
 * =========================================================
 */


export async function loadLatestExperienceVersion(
    skillId
){

    const history =

        await loadExperienceHistory(
            skillId
        );




    if(
        history.length === 0
    ){

        return null;

    }




    return history[0];

}
