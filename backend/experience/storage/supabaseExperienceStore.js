import {
    getSupabaseClient
} from "../../storage/supabaseClient.js";



/*
 * =========================================================
 * JESSICA SUPABASE EXPERIENCE STORE
 * =========================================================
 *
 * Работа с активной памятью Experience.
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
 * - сохраняет новые версии;
 * - создаёт Skill;
 * - работает с Learning.
 *
 * =========================================================
 */



const EXPERIENCE_TABLE =
    "jessica_experience_skills";





/*
 * =========================================================
 * NORMALIZE
 * =========================================================
 */


function normalizeArray(
    value
) {

    return Array.isArray(value)
        ? value
            .filter(Boolean)
        : [];

}



function normalizeSkill(
    value
) {


    if (
        !value ||
        typeof value !== "object"
    ) {

        return null;

    }



    const id =
        String(
            value.id ||
            value.skillId ||
            ""
        )
        .trim();



    if (!id) {

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



        workflow:
            Array.isArray(
                value.workflow
            )
            ?
            value.workflow
            :
            [],



        triggerPatterns:
            normalizeArray(
                value.triggerPatterns
            ),



        constraints:
            normalizeArray(
                value.constraints
            ),



        validationRules:
            normalizeArray(
                value.validationRules
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
            )

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
        .eq(
            "enabled",
            true
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
            `Experience load error: ${error.message}`
        );

    }



    if (
        !Array.isArray(data)
    ) {

        return [];

    }




    /*
     * Оставляем только последнюю версию каждого Skill
     */


    const latest =
        new Map();



    for (
        const row
        of data
    ) {


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



        if (
            !skill
        ) {

            continue;

        }



        const current =
            latest.get(
                skill.id
            );



        if (
            !current ||
            skill.version > current.version
        ) {

            latest.set(
                skill.id,
                skill
            );

        }


    }



    return Array.from(
        latest.values()
    );


}








/*
 * =========================================================
 * LOAD SINGLE SKILL
 * =========================================================
 */


export async function loadExperienceSkill(
    skillId
) {


    const id =
        String(
            skillId || ""
        )
        .trim();



    if (
        !id
    ) {

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
) {


    const id =
        String(
            skillId || ""
        )
        .trim();



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




    if (
        error
    ) {

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
