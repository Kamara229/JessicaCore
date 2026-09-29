import {
    getSupabaseClient
} from "../../storage/supabaseClient.js";



/*
 * =========================================================
 * JESSICA SUPABASE EXPERIENCE STORE v3
 * =========================================================
 *
 * Активная память Experience.
 *
 *
 * Отвечает:
 *
 * - загрузка актуальных Skills;
 * - получение Skill;
 * - отключение Skill.
 *
 *
 * НЕ:
 *
 * - обучает;
 * - создаёт версии;
 * - сохраняет Skill.
 *
 * =========================================================
 */



const EXPERIENCE_TABLE =
    "jessica_experience_skills";









function normalizeArray(
    value
){

    return Array.isArray(value)

        ?

        value.filter(Boolean)

        :

        [];

}









function normalizeSkill(
    value
){

    if(
        !value ||
        typeof value !== "object"
    ){

        return null;

    }



    const id =

        String(

            value.id ||

            value.skillId ||

            ""

        )
        .trim();





    if(
        !id
    ){

        return null;

    }







    return {


        ...value,



        id,



        version:

            Number(
                value.version || 1
            ),



        enabled:

            value.enabled !== false,



        confidence:

            Number(
                value.confidence || 0
            ),







        category:

            value.category ||
            "general",







        workflow:

            normalizeArray(
                value.workflow
            ),



        keywords:

            normalizeArray(
                value.keywords
            ),



        tags:

            normalizeArray(
                value.tags
            ),



        triggerPatterns:

            normalizeArray(
                value.triggerPatterns
            ),



        validationRules:

            normalizeArray(
                value.validationRules
            ),



        constraints:

            normalizeArray(
                value.constraints
            ),



        strategy:

            normalizeArray(
                value.strategy
            ),



        sourcePriority:

            normalizeArray(
                value.sourcePriority
            ),



        successfulPatterns:

            normalizeArray(
                value.successfulPatterns
            ),



        failurePatterns:

            normalizeArray(
                value.failurePatterns
            ),



        avoidPatterns:

            normalizeArray(
                value.avoidPatterns
            ),







        statistics:

        {


            successfulRuns:

                Number(

                    value.statistics
                    ?.successfulRuns

                    ||

                    0

                ),



            failedRuns:

                Number(

                    value.statistics
                    ?.failedRuns

                    ||

                    0

                ),



            lastUsedAt:

                value.statistics
                ?.lastUsedAt

                ||

                null,



            lastResult:

                value.statistics
                ?.lastResult

                ||

                null


        },







        metadata:

        {

            ...(value.metadata || {})

        }


    };

}









/*
 * =========================================================
 * LOAD ACTIVE EXPERIENCE
 * =========================================================
 */


export async function loadExperiences()
{


    const supabase =

        getSupabaseClient();





    const {
        data,
        error
    }

    =

    await supabase

        .from(
            EXPERIENCE_TABLE
        )

        .select(
            `
            id,
            payload,
            enabled,
            version
            `
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

            `Experience load error: ${error.message}`

        );

    }








    if(
        !Array.isArray(data)
    ){

        return [];

    }








    const latest =
        new Map();









    for(
        const row of data
    ){


        const skill =

            normalizeSkill({

                ...(row.payload || {}),



                id:

                    row.id,



                version:

                    row.version,



                enabled:

                    row.enabled


            });





        if(
            !skill
        ){

            continue;

        }








        const current =

            latest.get(
                skill.id
            );





        if(
            !current
            ||
            skill.version > current.version
        ){

            latest.set(

                skill.id,

                skill

            );

        }


    }









    return Array.from(

        latest.values()

    )

    .filter(

        skill =>

            skill.enabled === true

    );


}









/*
 * =========================================================
 * LOAD SINGLE
 * =========================================================
 */


export async function loadExperienceSkill(
    skillId
){

    const id =

        String(
            skillId || ""
        )
        .trim();





    if(
        !id
    ){

        return null;

    }







    const skills =

        await loadExperiences();





    return skills.find(

        skill =>

            skill.id === id

    )
    ||
    null;


}









/*
 * =========================================================
 * DISABLE
 * =========================================================
 */


export async function disableExperience(
    skillId
){

    const id =

        String(
            skillId || ""
        )
        .trim();





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
            EXPERIENCE_TABLE
        )

        .update({

            enabled:false,


            updated_at:

                new Date()
                .toISOString()


        })

        .eq(
            "id",
            id
        )

        .select(
            "id"
        );









    if(
        error
    ){

        throw new Error(

            `Disable Experience error: ${error.message}`

        );

    }







    return {


        success:

            Array.isArray(data)

            &&

            data.length > 0,



        id


    };

}
