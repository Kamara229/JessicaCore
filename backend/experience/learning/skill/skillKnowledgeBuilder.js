/*
 * =========================================================
 * JESSICA EXPERIENCE SKILL KNOWLEDGE BUILDER
 * =========================================================
 *
 * Proposed Experience
 *        ↓
 * Canonical Skill Knowledge
 *
 *
 * Отвечает только за содержимое навыка:
 *
 * - workflow;
 * - triggers;
 * - validation;
 * - constraints;
 * - tools;
 * - patterns;
 * - examples.
 *
 * =========================================================
 */


import {
    normalizeStringArray,
    normalizeObjectArray
} from "./skillUtils.js";


export function buildSkillKnowledge(
    proposedExperience
) {

    return {


        workflow:

            normalizeStringArray(
                proposedExperience?.workflow
            ),



        triggerPatterns:

            normalizeStringArray(
                proposedExperience?.triggerPatterns
            ),



        keywords:

            normalizeStringArray(
                proposedExperience?.keywords
            ),



        tags:

            normalizeStringArray(
                proposedExperience?.tags
            ),



        validationRules:

            normalizeStringArray(
                proposedExperience?.validationRules
            ),



        constraints:

            normalizeStringArray(
                proposedExperience?.constraints
            ),



        strategy:

            normalizeStringArray(
                proposedExperience?.strategy
            ),



        sourcePriority:

            normalizeStringArray(
                proposedExperience?.sourcePriority
            ),



        /*
         * Раньше это поле терялось
         * в Learning Skill Builder.
         */


        requiredTools:

            normalizeStringArray(
                proposedExperience?.requiredTools
            ),



        successfulPatterns:

            normalizeStringArray(
                proposedExperience?.successfulPatterns
            ),



        failurePatterns:

            normalizeStringArray(
                proposedExperience?.failurePatterns
            ),



        avoidPatterns:

            normalizeStringArray(
                proposedExperience?.avoidPatterns
            ),



        examples:

            normalizeObjectArray(
                proposedExperience?.examples
            )

    };

}
