/*
 * =========================================================
 * JESSICA LEARNING SKILL IDENTITY
 * =========================================================
 */

export function getLearningSkillId(
    skill
) {

    if(
        typeof skill === "string"
    ){

        return skill.trim();

    }


    return String(

        skill?.id

        ||

        skill?.skillId

        ||

        skill?.name

        ||

        ""

    )
        .trim();

}


export function uniqueLearningSkills(
    skills
) {

    const source =

        Array.isArray(
            skills
        )

            ? skills

            : [];


    const result = [];

    const seen =
        new Set();


    for(
        const skill
        of source
    ){

        const id =

            getLearningSkillId(
                skill
            );


        if(
            !id
            ||
            seen.has(
                id
            )
        ){

            continue;

        }


        seen.add(
            id
        );


        result.push(
            skill
        );

    }


    return result;

}
