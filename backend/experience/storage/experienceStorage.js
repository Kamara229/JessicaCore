/*
 * =========================================================
 * JESSICA EXPERIENCE STORAGE v2
 * =========================================================
 *
 * Единый слой хранения Experience Skills.
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
 * - сохранение новых версий;
 * - история;
 * - получение версии;
 * - отключение Skill.
 *
 *
 * НЕ:
 *
 * - обучает;
 * - принимает решение;
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
 * NORMALIZE HELPERS
 * =========================================================
 */


function normalizeText(
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

        return 1;

    }


    return version;

}





function normalizeConfidence(
    value
){

    const confidence =
        Number(value);


    if(
        !Number.isFinite(confidence)
    ){

        return 0;

    }


    return Math.max(

        0,

        Math.min(

            1,

            confidence

        )

    );

}









function normalizeExperience(
    experience
){

    if(
        !experience ||
        typeof experience !== "object"
    ){

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

            normalizeConfidence(
                experience.confidence
            ),



        metadata:

            {

                ...(experience.metadata || {})

            }

    };

}









/*
 * =========================================================
 * LOAD ACTIVE EXPERIENCE
 * =========================================================
 */


export async function loadExperienceSkills()
{

    try{


        const skills =

            await loadExperiences();



        return Array.isArray(skills)

            ?

            skills

            :

            [];



    }catch(error){


        console.error(

            "Jessica Experience load error:",

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
){

    const normalized =

        normalizeExperience(
            experience
        );





    if(
        !normalized
    ){

        return {

            success:false,

            error:
                "Experience Skill отсутствует"

        };

    }







    if(
        !normalized.id
    ){

        return {

            success:false,

            error:
                "Experience Skill ID отсутствует"

        };

    }







    if(
        !normalized.name
    ){

        return {

            success:false,

            error:
                "Experience Skill name отсутствует"

        };

    }







    try{



        /*
         * =================================================
         * VERSION CONTROL
         * =================================================
         */


        const history =

            await getExperienceHistory(
                normalized.id
            );





        const duplicate =

            history.some(

                item =>

                    Number(
                        item.version
                    )
                    ===
                    Number(
                        normalized.version
                    )

            );





        if(
            duplicate
        ){

            return {


                success:false,


                error:

                    "Такая версия Experience уже существует",


                skillId:

                    normalized.id,


                version:

                    normalized.version


            };

        }









        /*
         * =================================================
         * AUTONOMOUS LEARNING META
         * =================================================
         */


        normalized.metadata = {


            ...normalized.metadata,



            storage:

                "experience-storage",



            storedAt:

                new Date()
                    .toISOString(),



            storageVersion:

                "v2"

        };









        /*
         * =================================================
         * ATOMIC SAVE
         * =================================================
         */


        const result =

            await saveExperienceAtomic(
                normalized
            );





        if(
            !result?.success
        ){

            return {


                success:false,


                error:

                    result?.error ||

                    "Ошибка сохранения Experience"


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





    }catch(error){


        console.error(

            "Jessica Experience save error:",

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
 * DISABLE EXPERIENCE
 * =========================================================
 */


export async function disableExperienceSkill(
    skillId
){

    const id =
        normalizeText(
            skillId
        );



    if(
        !id
    ){

        return {

            success:false,

            error:
                "Skill ID отсутствует"

        };

    }



    return await disableExperience(
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
){

    const id =
        normalizeText(
            skillId
        );



    if(
        !id
    ){

        return [];

    }



    try{


        const history =

            await loadExperienceHistory(
                id
            );



        return Array.isArray(history)

            ?

            history

            :

            [];



    }catch(error){


        console.error(

            "Jessica Experience history error:",

            error

        );



        return [];

    }

}









/*
 * =========================================================
 * VERSION
 * =========================================================
 */


export async function getExperienceVersion(
    skillId,
    version
){

    const id =
        normalizeText(
            skillId
        );



    if(
        !id
    ){

        return null;

    }



    try{


        return await loadExperienceVersion(

            id,

            normalizeVersion(
                version
            )

        );



    }catch(error){


        console.error(

            "Jessica Experience version error:",

            error

        );



        return null;

    }

}









/*
 * =========================================================
 * LATEST VERSION
 * =========================================================
 */


export async function getLatestExperienceVersion(
    skillId
){

    const history =

        await getExperienceHistory(
            skillId
        );



    if(
        history.length === 0
    ){

        return null;

    }






    return history.sort(

        (a,b)=>

            Number(
                b.version || 0
            )

            -

            Number(
                a.version || 0
            )

    )[0];

        }
