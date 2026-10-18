/*
 * =========================================================
 * JESSICA SKILL TRACE VIEW
 * =========================================================
 *
 * Формирует узкий Learning Trace
 * для одного реально применённого Skill.
 *
 * Это важно для complex execution:
 * каждый Skill должен обучаться только
 * на своей подзадаче и своих tools.
 * =========================================================
 */


import {
    getLearningSkillId
} from "./skillIdentity.js";


function safeArray(
    value
) {

    return Array.isArray(value)

        ? value

        : [];

}


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


function getResultExperience(
    result
) {

    return isObject(
        result?.experience
    )

        ? result.experience

        : {};

}


function resultUsesSkill(
    result,
    skillId
) {

    const experience =

        getResultExperience(
            result
        );


    if(
        experience.used !== true
    ){

        return false;

    }


    return safeArray(
        experience.skills
    )
        .some(
            skill =>
                getLearningSkillId(
                    skill
                ) ===
                skillId
        );

}


function collectResultTools(
    result
) {

    const tools =

        safeArray(
            result
                ?.executionMeta
                ?.usedTools
        );


    return [
        ...new Set(
            tools
                .map(
                    item =>
                        String(
                            item || ""
                        )
                            .trim()
                )
                .filter(Boolean)
        )
    ];

}


function buildExperienceUsage(
    trace,
    result,
    skill
) {

    const resultExperience =

        getResultExperience(
            result
        );


    const skillId =

        getLearningSkillId(
            skill
        );


    return {

        used:
            true,

        found:
            true,

        source:

            resultExperience.source

            ||

            trace
                ?.experienceUsage
                ?.source

            ||

            null,

        confidence:

            Number(

                resultExperience.confidence

                ??

                trace
                    ?.experienceUsage
                    ?.confidence

                ??

                0

            )
            || 0,

        skills: [
            skill
        ],

        skillIds:

            skillId

                ? [
                    skillId
                ]

                : []

    };

}


function buildChildTrace(
    trace,
    result,
    skill
) {

    const success =

        result?.success === true

        ||

        String(
            result?.status || ""
        )
            .trim()
            .toUpperCase()
        ===
        "COMPLETED";


    return {

        ...trace,

        task:

            String(

                result?.task

                ||

                trace?.task

                ||

                ""

            )
                .trim(),

        status:

            result?.status

            ||

            (
                success

                    ? "COMPLETED"

                    : "FAILED"
            ),

        result: {
            ...result
        },

        validation:

            result?.validation

            ||

            null,

        terminal:

            result?.terminal

            ||

            null,

        statistics: {

            ...(trace?.statistics || {}),

            completed:

                success

                    ? 1

                    : 0,

            failed:

                success

                    ? 0

                    : 1

        },

        stats: {

            ...(trace?.stats || {}),

            completed:

                success

                    ? 1

                    : 0,

            failed:

                success

                    ? 0

                    : 1

        },

        experienceUsage:

            buildExperienceUsage(
                trace,
                result,
                skill
            ),

        experience:

            buildExperienceUsage(
                trace,
                result,
                skill
            ),

        executedTools:

            collectResultTools(
                result
            ),

        failures:

            result?.failure

                ? [
                    result.failure
                ]

                : [],

        steps:

            safeArray(
                result
                    ?.executionMeta
                    ?.stepsHistory
            ),

        events:
            []

    };

}


/*
 * =========================================================
 * PUBLIC
 * =========================================================
 */


export function buildSkillTraceView(
    trace,
    skill
) {

    const skillId =

        getLearningSkillId(
            skill
        );


    if(
        !trace
        ||
        !skillId
    ){

        return null;

    }


    const childResults =

        safeArray(
            trace
                ?.result
                ?.results
        );


    const matched =

        childResults.filter(
            result =>
                resultUsesSkill(
                    result,
                    skillId
                )
        );


    if(
        matched.length > 0
    ){

        /*
         * Один Skill может использоваться
         * несколькими child executions.
         *
         * Для текущего Learning Evidence
         * берём последний подтверждённый child,
         * а occurrence/history остаются
         * внутри самого Skill.
         */


        return buildChildTrace(

            trace,

            matched[
                matched.length - 1
            ],

            skill

        );

    }


    /*
     * Single-task / legacy fallback.
     */


    return {

        ...trace,

        experienceUsage: {

            ...(trace?.experienceUsage || {}),

            used:
                true,

            found:
                true,

            skills: [
                skill
            ],

            skillIds: [
                skillId
            ]

        },

        experience: {

            ...(trace?.experienceUsage || {}),

            used:
                true,

            found:
                true,

            skills: [
                skill
            ],

            skillIds: [
                skillId
            ]

        }

    };

}
