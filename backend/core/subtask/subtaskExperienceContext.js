/*
 * =========================================================
 * JESSICA SUBTASK EXPERIENCE CONTEXT v1
 * =========================================================
 *
 * Адаптирует результат Experience Resolver
 * к каноническому контракту Execution Context.
 *
 *
 * Resolver:
 *
 * {
 *   found,
 *   source,
 *   confidence,
 *   experience,
 *   planningContext
 * }
 *
 *        ↓
 *
 * Execution:
 *
 * {
 *   used,
 *   found,
 *   source,
 *   confidence,
 *   skills,
 *   context
 * }
 *
 *
 * НЕ:
 *
 * - ищет Experience;
 * - изменяет Skill;
 * - принимает Learning Decision;
 * - работает с Supabase.
 *
 * =========================================================
 */


function isObject(
    value
) {

    return Boolean(

        value
        &&
        typeof value === "object"
        &&
        !Array.isArray(value)

    );

}


function normalizeText(
    value
) {

    return String(
        value || ""
    )
    .trim();

}


function normalizeNumber(
    value
) {

    const number =
        Number(value);


    return Number.isFinite(number)

        ? number

        : 0;

}


/*
 * =========================================================
 * SKILL KEY
 * =========================================================
 */


function buildSkillKey(
    skill
) {

    if(
        !isObject(skill)
    ){

        return "";

    }


    const id =

        normalizeText(

            skill.id

            ||

            skill.skillId

        );


    if(
        id
    ){

        return `id:${id}`;

    }


    const name =

        normalizeText(
            skill.name
        );


    if(
        name
    ){

        return `name:${name.toLowerCase()}`;

    }


    return "";

}


/*
 * =========================================================
 * COLLECT SKILLS
 * =========================================================
 */


function collectExperienceSkills(
    experienceResult
) {

    const result = [];

    const keys =
        new Set();


    const pushSkill = (
        skill
    ) => {

        if(
            !isObject(skill)
        ){

            return;

        }


        const key =

            buildSkillKey(
                skill
            );


        if(
            key
            &&
            keys.has(key)
        ){

            return;

        }


        if(
            key
        ){

            keys.add(
                key
            );

        }


        result.push({
            ...skill
        });

    };


    /*
     * Уже канонический формат.
     */


    if(
        Array.isArray(
            experienceResult?.skills
        )
    ){

        for(
            const skill
            of experienceResult.skills
        ){

            pushSkill(
                skill
            );

        }

    }


    /*
     * Experience Resolver:
     *
     * result.experience.skills
     */


    if(
        Array.isArray(
            experienceResult
                ?.experience
                ?.skills
        )
    ){

        for(
            const skill
            of experienceResult.experience.skills
        ){

            pushSkill(
                skill
            );

        }

    }


    /*
     * Некоторые Planning Context
     * могут содержать skills.
     */


    if(
        Array.isArray(
            experienceResult
                ?.planningContext
                ?.skills
        )
    ){

        for(
            const skill
            of experienceResult.planningContext.skills
        ){

            pushSkill(
                skill
            );

        }

    }


    /*
     * Или один выбранный Experience.
     */


    const planningExperience =

        experienceResult
            ?.planningContext
            ?.experience;


    if(
        isObject(
            planningExperience
        )
    ){

        if(
            Array.isArray(
                planningExperience.skills
            )
        ){

            for(
                const skill
                of planningExperience.skills
            ){

                pushSkill(
                    skill
                );

            }

        }else{


            const hasIdentity =

                normalizeText(
                    planningExperience.id
                )

                ||

                normalizeText(
                    planningExperience.skillId
                )

                ||

                normalizeText(
                    planningExperience.name
                );


            if(
                hasIdentity
            ){

                pushSkill(
                    planningExperience
                );

            }

        }

    }


    return result;

}


/*
 * =========================================================
 * BUILD EXECUTION EXPERIENCE
 * =========================================================
 */


export function buildExecutionExperienceContext(
    experienceResult
) {

    if(
        !isObject(
            experienceResult
        )
    ){

        return {

            used:
                false,

            found:
                false,

            source:
                null,

            confidence:
                0,

            skills:
                [],

            context:
                null

        };

    }


    const found =

        experienceResult.found === true;


    const skills =

        collectExperienceSkills(
            experienceResult
        );


    /*
     * Здесь used означает:
     *
     * Experience был передан Planner
     * и Execution Context.
     *
     * Это не утверждение,
     * что каждый элемент Skill
     * причинно повлиял на ответ.
     */


    const used =

        experienceResult.used === true

        ||

        found;


    return {

        used,

        found,

        source:

            normalizeText(
                experienceResult.source
            )

            ||

            null,

        confidence:

            normalizeNumber(
                experienceResult.confidence
            ),

        skills,

        context:

            experienceResult.planningContext

            ??

            experienceResult.experience

            ??

            experienceResult.context

            ??

            null

    };

    }
