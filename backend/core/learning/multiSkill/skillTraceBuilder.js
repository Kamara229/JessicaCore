/*
 * =========================================================
 * JESSICA SKILL TRACE BUILDER
 * =========================================================
 */


import {
    getLearningSkillId
} from "./skillIdentity.js";


import {
    collectResultTools,
    getResultExperience,
    safeLearningArray
} from "./skillTraceResult.js";


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

        // Carry independent, per-child application proof into Learning.
        applied:
            resultExperience.applied === true &&
            Array.isArray(resultExperience.appliedSkillIds) &&
            resultExperience.appliedSkillIds.includes(skillId),

        appliedSkillIds:
            resultExperience.applied === true &&
            Array.isArray(resultExperience.appliedSkillIds) &&
            resultExperience.appliedSkillIds.includes(skillId)
                ? [skillId] : [],

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


export function buildChildSkillTrace(
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


    const experienceUsage =

        buildExperienceUsage(
            trace,
            result,
            skill
        );


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

        experienceUsage,

        experience: {
            ...experienceUsage
        },

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

            safeLearningArray(
                result
                    ?.executionMeta
                    ?.stepsHistory
            ),

        events:
            []

    };

}
