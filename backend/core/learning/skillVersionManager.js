/*
 * =========================================================
 * JESSICA SKILL VERSION MANAGER
 * =========================================================
 *
 * Управляет версиями Experience Skill.
 *
 *
 * НЕ:
 *
 * - пишет в Supabase;
 * - вызывает AI;
 * - принимает решение об обучении.
 *
 * Только определяет:
 *
 * create / update version
 *
 * =========================================================
 */



/*
 * =========================================================
 * CREATE NEW SKILL
 * =========================================================
 */


function createNewSkill(
    experience
) {

    return {

        action:
            "CREATE",


        version:
            1,


        skill:

            {

                ...experience,

                version:
                    1

            }


    };

}





/*
 * =========================================================
 * UPDATE EXISTING SKILL
 * =========================================================
 */


function updateSkill(
    existing,
    experience
) {


    const newVersion =
        Number(
            existing.version || 1
        )
        +
        1;



    return {


        action:
            "UPDATE",



        version:
            newVersion,



        skill:
        {

            ...existing,


            version:
                newVersion,



            confidence:
                Math.min(

                    (
                        Number(existing.confidence || 0)
                        +
                        Number(experience.confidence || 0)
                    )
                    /
                    2,

                    1

                ),



            examples:

                [

                    ...(existing.examples || []),

                    ...(experience.examples || [])

                ]

        }


    };

}





/*
 * =========================================================
 * RESOLVE VERSION
 * =========================================================
 */


export function resolveSkillVersion(
    existingSkill,
    newExperience
) {


    /*
     * Новый Skill
     */


    if (
        !existingSkill
    ) {

        return createNewSkill(
            newExperience
        );

    }



    /*
     * Обновление
     */


    return updateSkill(

        existingSkill,

        newExperience

    );


}
