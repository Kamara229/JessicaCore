import {
    getSupabaseClient
} from "../../storage/supabaseClient.js";



/*
 * =========================================================
 * JESSICA EXPERIENCE WRITER
 * =========================================================
 *
 * Атомарное сохранение Experience Skill.
 *
 *
 * Flow:
 *
 * Experience Skill
 *        ↓
 * Writer
 *        ↓
 * PostgreSQL RPC
 *        ↓
 * History + Current Skill
 *
 *
 * НЕ:
 *
 * - анализирует обучение;
 * - создаёт Skill;
 * - ищет версии;
 * - работает с Planner.
 *
 * =========================================================
 */



const SAVE_SKILL_RPC =
    "save_jessica_experience_skill";





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

        return null;

    }


    return version;

}





function normalizeBoolean(
    value
) {

    return value !== false;

}





/*
 * =========================================================
 * SAVE EXPERIENCE ATOMIC
 * =========================================================
 */


export async function saveExperienceAtomic(
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





    const skillId =
        normalizeText(
            experience.id ||
            experience.skillId
        );



    const version =
        normalizeVersion(
            experience.version
        );



    if (
        !skillId
    ) {

        throw new Error(
            "Skill ID отсутствует"
        );

    }



    if (
        !version
    ) {

        throw new Error(
            "Версия Skill некорректна"
        );

    }






    const payload = {

        ...experience,


        id:
            skillId,


        version,


        enabled:
            normalizeBoolean(
                experience.enabled
            )

    };





    const metadata = {


        ...(experience.metadata || {}),


        previousVersion:
            experience.previousVersion ||
            null,


        mode:
            experience.mode ||
            "create"

    };







    const supabase =
        getSupabaseClient();





    const {
        data,
        error
    } =
        await supabase.rpc(

            SAVE_SKILL_RPC,

            {


                p_skill_id:
                    skillId,


                p_version:
                    version,


                p_previous_version:
                    experience.previousVersion ||
                    null,


                p_mode:
                    metadata.mode,


                p_enabled:
                    payload.enabled,



                p_payload:
                    payload,



                p_metadata:
                    metadata


            }

        );







    if (
        error
    ) {


        throw new Error(

            "Experience atomic save failed: "

            +

            error.message

        );

    }







    const rpcResult =
        Array.isArray(data)
            ? data[0]
            : data;






    const success =

        rpcResult?.success === true

        ||

        rpcResult?.success === "true";







    if (
        !success
    ) {


        return {


            success:false,


            error:

                rpcResult?.error

                ||

                "RPC не подтвердил сохранение"


        };

    }







    return {


        success:true,


        id:
            rpcResult?.id ||
            skillId,



        skillId,



        version:
            Number(

                rpcResult?.version

                ||

                version

            ),



        historyId:
            rpcResult?.historyId ||
            null,



        enabled:
            rpcResult?.enabled !== false,



        experience:
            payload


    };


}
