import {
    getSupabaseClient
} from "../../storage/supabaseClient.js";



/*
 * =========================================================
 * JESSICA EXPERIENCE WRITER v4
 * =========================================================
 *
 * Атомарное сохранение Experience Skill.
 *
 *
 * Experience Skill
 *        ↓
 * Writer
 *        ↓
 * PostgreSQL RPC
 *        ↓
 * Current + History
 *
 *
 * НЕ:
 *
 * - анализирует обучение;
 * - принимает решения;
 * - вызывает AI.
 *
 * =========================================================
 */



const SAVE_SKILL_RPC =
    "save_jessica_experience_skill";







function normalizeText(value)
{

    return String(
        value || ""
    )
    .trim();

}







function normalizeVersion(value)
{

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







function normalizeBoolean(value)
{

    return value !== false;

}







function normalizePreviousVersion(value)
{

    if(
        value === null
        ||
        value === undefined
    ){

        return null;

    }


    return normalizeVersion(
        value
    );

}







function validateVersionChain(
    experience
){

    const version =
        normalizeVersion(
            experience.version
        );


    const previous =
        normalizePreviousVersion(
            experience.previousVersion
        );



    if(
        version === 1
    ){

        return previous === null;

    }



    if(
        previous === null
    ){

        return false;

    }



    return (

        version === previous + 1

    );

}









function buildMetadata(
    experience
){

    return {


        ...(experience.metadata || {}),



        previousVersion:

            normalizePreviousVersion(
                experience.previousVersion
            ),



        mode:

            experience.mode
            ||
            "create",



        learningMode:

            experience.metadata?.learningMode
            ||
            "autonomous",



        createdBy:

            experience.metadata?.createdBy
            ||
            "jessica-learning",



        storage:

            "experience-writer",



        storageVersion:

            "v4",



        storedAt:

            new Date()
                .toISOString()


    };

}









export async function saveExperienceAtomic(
    experience
){

    if(
        !experience
        ||
        typeof experience !== "object"
    ){

        throw new Error(
            "Experience отсутствует"
        );

    }








    const skillId =

        normalizeText(

            experience.id

            ||

            experience.skillId

        );




    if(
        !skillId
    ){

        throw new Error(
            "Skill ID отсутствует"
        );

    }







    const version =

        normalizeVersion(
            experience.version
        );




    if(
        !version
    ){

        throw new Error(
            "Версия Skill некорректна"
        );

    }







    if(
        !validateVersionChain(
            experience
        )
    ){

        throw new Error(
            "Нарушена цепочка версий Experience"
        );

    }








    const previousVersion =

        normalizePreviousVersion(
            experience.previousVersion
        );









    const payload = {


        ...experience,



        id:

            skillId,



        version,



        previousVersion,



        enabled:

            normalizeBoolean(
                experience.enabled
            )



    };









    const metadata =

        buildMetadata(
            payload
        );









    const supabase =

        getSupabaseClient();









    const {
        data,
        error
    }

    =

    await supabase.rpc(

        SAVE_SKILL_RPC,

        {


            p_skill_id:

                skillId,



            p_version:

                version,



            p_previous_version:

                previousVersion,



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









    if(
        error
    ){

        throw new Error(

            "Experience atomic save failed: "

            +

            error.message

        );

    }









    const rpcResult =

        Array.isArray(data)

        ?

        data[0]

        :

        data;









    const success =

        rpcResult?.success === true

        ||

        rpcResult?.success === "true";









    if(
        !success
    ){

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

            rpcResult?.id

            ||

            skillId,



        skillId,



        version:

            Number(

                rpcResult?.version

                ||

                version

            ),



        historyId:

            rpcResult?.historyId

            ||

            null,



        enabled:

            rpcResult?.enabled !== false,



        experience:

            payload



    };


}
