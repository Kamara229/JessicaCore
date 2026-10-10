/*
 * =========================================================
 * JESSICA SUBTASK EXPERIENCE CONTEXT v2
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
 *   experience,        // FULL canonical Skill
 *   planningContext    // compact Planner DTO
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
 *   skills,            // FULL canonical Skills
 *   context            // compact PlanningContext
 * }
 *
 *
 * КРИТИЧЕСКОЕ ПРАВИЛО:
 *
 * experienceResult.experience
 *
 * и
 *
 * experienceResult.planningContext.experience
 *
 * — это НЕ один и тот же контракт.
 *
 *
 * Первый:
 *
 * полный Experience Skill,
 * необходимый Runtime / Trace / Learning.
 *
 *
 * Второй:
 *
 * сокращённый DTO для AI Planner.
 *
 *
 * Planner DTO нельзя использовать
 * как источник Skill для Learning,
 * иначе теряются:
 *
 * - examples;
 * - learning;
 * - statistics;
 * - requiredTools;
 * - metadata;
 * - другие накопленные знания.
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


/*
 * =========================================================
 * OBJECT
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


/*
 * =========================================================
 * NORMALIZE TEXT
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


/*
 * =========================================================
 * NORMALIZE NUMBER
 * =========================================================
 */


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
 * SKILL IDENTITY
 * =========================================================
 */


function hasSkillIdentity(
    skill
) {

    if(
        !isObject(skill)
    ){

        return false;

    }


    return Boolean(

        normalizeText(
            skill.id
        )

        ||

        normalizeText(
            skill.skillId
        )

        ||

        normalizeText(
            skill.name
        )

    );

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

        return (

            "name:"

            +

            name.toLowerCase()

        );

    }


    return "";

}


/*
 * =========================================================
 * COLLECT SKILLS
 * =========================================================
 *
 * ПОРЯДОК ИМЕЕТ ЗНАЧЕНИЕ.
 *
 *
 * 1. FULL Resolver Skill
 *
 *    experienceResult.experience
 *
 *
 * 2. Уже канонический skills[]
 *
 *
 * 3. Legacy wrapper experience.skills[]
 *
 *
 * 4. PlanningContext fallbacks
 *
 *
 * Благодаря дедупликации по Skill ID
 * полный Resolver Skill всегда имеет
 * приоритет над Planner DTO.
 *
 * =========================================================
 */


function collectExperienceSkills(
    experienceResult
) {

    const result = [];

    const keys =
        new Set();


    /*
     * =====================================================
     * PUSH
     * =====================================================
     */


    const pushSkill = (
        skill
    ) => {

        if(
            !isObject(skill)
            ||
            !hasSkillIdentity(skill)
        ){

            return;

        }


        const key =

            buildSkillKey(
                skill
            );


        /*
         * Skill с тем же ID уже сохранён.
         *
         * Так как FULL Skill собирается первым,
         * компактный Planner DTO позже
         * не сможет его перезаписать.
         */


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


        /*
         * Только shallow copy.
         *
         * Мы не нормализуем Skill здесь,
         * потому что полный canonical payload
         * должен дойти до Learning без потерь.
         */


        result.push({

            ...skill

        });

    };


    /*
     * =====================================================
     * 1. FULL RESOLVER EXPERIENCE
     * =====================================================
     *
     * Текущий Experience Core возвращает:
     *
     * {
     *   found: true,
     *   experience: FULL_SKILL,
     *   planningContext: ...
     * }
     *
     *
     * Это главный источник для Learning.
     * =====================================================
     */


    const resolverExperience =

        experienceResult
            ?.experience;


    if(
        isObject(
            resolverExperience
        )
    ){

        /*
         * Текущий формат:
         *
         * experience = Skill
         */


        if(
            hasSkillIdentity(
                resolverExperience
            )
        ){

            pushSkill(
                resolverExperience
            );

        }


        /*
         * Legacy compatibility:
         *
         * experience = {
         *     skills:[...]
         * }
         */


        if(
            Array.isArray(
                resolverExperience.skills
            )
        ){

            for(
                const skill
                of resolverExperience.skills
            ){

                pushSkill(
                    skill
                );

            }

        }

    }


    /*
     * =====================================================
     * 2. CANONICAL TOP-LEVEL SKILLS
     * =====================================================
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
     * =====================================================
     * 3. PLANNING CONTEXT SKILLS
     * =====================================================
     *
     * Compatibility fallback.
     *
     * Они могут быть сокращёнными,
     * поэтому идут ТОЛЬКО после
     * resolverExperience.
     * =====================================================
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
            of experienceResult
                .planningContext
                .skills
        ){

            pushSkill(
                skill
            );

        }

    }


    /*
     * =====================================================
     * 4. PLANNER EXPERIENCE DTO
     * =====================================================
     *
     * Последний fallback.
     *
     * В нормальном современном flow
     * он НЕ должен становиться источником
     * Learning Skill, потому что FULL Skill
     * уже был добавлен выше.
     * =====================================================
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

        }else if(
            hasSkillIdentity(
                planningExperience
            )
        ){

            pushSkill(
                planningExperience
            );

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

    /*
     * =====================================================
     * INVALID RESULT
     * =====================================================
     */


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


    /*
     * =====================================================
     * FOUND
     * =====================================================
     */


    const found =

        experienceResult.found === true;


    /*
     * =====================================================
     * FULL SKILLS
     * =====================================================
     */


    const skills =

        collectExperienceSkills(
            experienceResult
        );


    /*
     * =====================================================
     * USED
     * =====================================================
     *
     * used означает:
     *
     * Experience был найден и передан
     * в Planning / Execution pipeline.
     *
     *
     * Это не утверждает,
     * что каждый отдельный элемент Skill
     * причинно повлиял на финальный ответ.
     *
     * =====================================================
     */


    const used =

        experienceResult.used === true

        ||

        (
            found
            &&
            skills.length > 0
        );


    /*
     * =====================================================
     * CONTEXT
     * =====================================================
     *
     * context — только Planner Context.
     *
     * FULL Skill сюда больше
     * намеренно НЕ подставляем.
     *
     * Полный Skill хранится в skills[].
     *
     * Это предотвращает смешивание:
     *
     * Planner DTO
     *
     * и
     *
     * Learning Skill.
     * =====================================================
     */


    const context =

        experienceResult.planningContext

        ??

        experienceResult.context

        ??

        null;


    /*
     * =====================================================
     * RESULT
     * =====================================================
     */


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

        context

    };

            }
