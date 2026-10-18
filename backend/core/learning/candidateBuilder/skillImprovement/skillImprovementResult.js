/*
 * =========================================================
 * JESSICA SKILL IMPROVEMENT RESULT
 * =========================================================
 */


import {
    normalizeArray,
    normalizeText,
    normalizeStringArray
} from "../candidateUtils.js";


import {
    resolveSkillExecutionProgram
} from "../../../../experience/execution/executionProgram.js";


import {
    buildSkillReference
} from "./skillReference.js";


export function buildSkillImprovementResult(
    context
) {

    if(!context){

        return null;

    }


    const {
        rawExistingSkill,
        existingSkill,
        examples,
        metrics,
        targetSkillId,
        toolKnowledge
    } = context;


    return {

        skills:

            [

                buildSkillReference(

                    rawExistingSkill,

                    existingSkill

                )

            ]
                .filter(Boolean),

        targetSkillId,

        baseVersion:

            Number(
                existingSkill.version || 0
            )

            ||

            null,

        name:

            normalizeText(
                existingSkill.name
            )

            ||

            (
                targetSkillId

                    ? `Jessica Skill ${targetSkillId}`

                    : "Jessica Experience Skill"
            ),

        category:

            normalizeText(
                existingSkill.category
            )

            ||

            "general",

        description:

            normalizeText(
                existingSkill.description
            ),

        workflow:

            normalizeArray(
                existingSkill.workflow
            ),

        validationRules:

            normalizeStringArray(
                existingSkill.validationRules
            ),

        /*
         * Trigger Patterns and Keywords are different signals.
         *
         * Autonomous Improvement must not promote broad
         * keywords such as "date" or "release" into triggers.
         */

        triggerPatterns:

            normalizeStringArray(
                existingSkill.triggerPatterns
            ),

        keywords:

            normalizeStringArray(
                existingSkill.keywords
            ),

        tags:

            normalizeStringArray(
                existingSkill.tags
            ),

        constraints:

            normalizeStringArray(
                existingSkill.constraints
            ),

        strategy:

            normalizeStringArray(
                existingSkill.strategy
            ),

        sourcePriority:

            normalizeStringArray(
                existingSkill.sourcePriority
            ),

        requiredTools:
            toolKnowledge.requiredTools,

        executionProgram:

            resolveSkillExecutionProgram(
                existingSkill
            ),

        successfulPatterns:

            normalizeStringArray(
                existingSkill.successfulPatterns
            ),

        failurePatterns:

            normalizeStringArray(
                existingSkill.failurePatterns
            ),

        avoidPatterns:

            normalizeStringArray(
                existingSkill.avoidPatterns
            ),

        examples,

        ...metrics,

        candidateType:
            "SKILL_IMPROVEMENT",

        improvementType:

            toolKnowledge.bootstrapped

                ? "TOOL_REQUIREMENT_BOOTSTRAP"

                : "EVIDENCE_REINFORCEMENT",

        source:
            "execution-trace"

    };

}
