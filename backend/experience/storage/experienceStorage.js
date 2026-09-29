/*
 * =========================================================
 * JESSICA EXPERIENCE STORAGE v3
 * =========================================================
 *
 * Единый слой хранения Experience Skills.
 *
 *
 * Отвечает:
 *
 * - загрузка Skills;
 * - сохранение версий;
 * - история;
 * - получение версии;
 * - отключение Skill.
 *
 *
 * НЕ:
 *
 * - обучает;
 * - принимает решения;
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

    const number =
        Number(value);


    if(
        !Number.isFinite(number)
    ){

        return 0;

    }


    return Math.max(
        0,
        Math.min(
            1,
            number
        )
    );

}









function normalizeArray(
    value
){

    if(
        !Array.isArray(value)
    ){

        return [];

    }


    return value

        .map(
            item =>
                normalizeText(item)
        )

        .filter(
            Boolean
        );

}









function normalizeExperience(
    experience
){

    if(
        !experience
        ||
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







        tags:

            normalizeArray(
                experience.tags
            ),



        triggerPatterns:

            normalizeArray(
                experience.triggerPatterns
            ),



        keywords:

            normalizeArray(
                experience.keywords
            ),



        validationRules:

            normalizeArray(
                experience.validationRules
            ),



        constraints:

            normalizeArray(
                experience.constraints
            ),



        metadata:

        {

            ...(experience.metadata || {})

        }

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



        return Array.isArray(skills)

            ?

            skills

            :

            [];


    }
    catch(error){


        console.error(

            "Experience load error:",

            error

        );


        return [];

    }

}









/*
 * =========================================================
 * SAVE EXPERIENCE
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
                "Experience отсутствует"

        };

    }





    if(
        !normalized.id
    ){

        return {

            success:false,

            error:
                "Skill ID отсутствует"

        };

    }





    if(
        !normalized.name
    ){

        return {

            success:false,

            error:
                "Skill name отсутствует"

        };

    }





    try {



        const history =

            await getExperienceHistory(
                normalized.id
            );





        const exists =

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
            exists
        ){

            return {


                success:false,


                error:
                    "Версия уже существует",


                skillId:
                    normalized.id,


                version:
                    normalized.version


            };

        }







        normalized.metadata = {


            ...normalized.metadata,


            storage:

                "experience-storage",



            storedAt:

                new Date()
                    .toISOString(),



            storageVersion:

                "v3"


        };








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

                    "Ошибка сохранения"


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



    }
    catch(error){


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
 * DISABLE
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


    }
    catch(error){


        console.error(

            "Experience history error:",

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


    }
    catch(error){


        console.error(

            "Experience version error:",

            error

        );


        return null;

    }

}









/*
 * =========================================================
 * LATEST
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
