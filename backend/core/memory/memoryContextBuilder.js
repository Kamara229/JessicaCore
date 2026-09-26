/*
 * =========================================================
 * JESSICA MEMORY CONTEXT BUILDER
 * =========================================================
 *
 * Формирует контекст из опыта Jessica.
 *
 *
 * Flow:
 *
 * Experience Skills
 *        ↓
 * Memory Context
 *        ↓
 * Planner
 *
 *
 * НЕ:
 *
 * - ищет Experience;
 * - изменяет Experience;
 * - создаёт Skills.
 *
 * =========================================================
 */





/*
 * =========================================================
 * NORMALIZE SKILL
 * =========================================================
 */


function normalizeSkill(
    skill
) {


    if (
        !skill ||
        typeof skill !== "object"
    ) {

        return null;

    }



    return {


        id:
            skill.id ||
            null,


        name:
            skill.name ||
            skill.skill_name ||
            "",



        description:
            skill.description ||
            "",



        workflow:
            Array.isArray(
                skill.workflow
            )
                ? skill.workflow
                : [],



        constraints:
            Array.isArray(
                skill.constraints
            )
                ? skill.constraints
                : [],



        examples:
            Array.isArray(
                skill.examples
            )
                ? skill.examples
                : [],



        confidence:
            Number(
                skill.confidence || 0
            )


    };

}





/*
 * =========================================================
 * BUILD CONTEXT
 * =========================================================
 */


export function buildMemoryContext(
    skills = []
) {


    const safeSkills =
        Array.isArray(
            skills
        )
            ? skills
            : [];




    const normalizedSkills =

        safeSkills

            .map(
                normalizeSkill
            )

            .filter(
                Boolean
            );





    return {


        hasExperience:
            normalizedSkills.length > 0,



        skills:
            normalizedSkills,



        skillCount:
            normalizedSkills.length,



        summary:

            normalizedSkills

                .map(
                    skill =>
                        skill.name
                )

                .filter(
                    Boolean
                )


    };


}
