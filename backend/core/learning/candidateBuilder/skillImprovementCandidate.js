/*
 * =========================================================
 * JESSICA SKILL IMPROVEMENT CANDIDATE BUILDER
 * =========================================================
 *
 * Existing Experience
 *        +
 * New Execution Evidence
 *        ↓
 * SKILL_IMPROVEMENT Candidate
 *
 *
 * Сейчас Improvement:
 *
 * - сохраняет Knowledge;
 * - добавляет Evidence;
 * - обновляет Learning Metrics;
 * - может восстановить requiredTools
 *   у legacy Skill с пустым контрактом.
 *
 *
 * Семантическое изменение workflow,
 * validationRules и constraints
 * здесь НЕ выполняется.
 *
 * =========================================================
 */


import {
    isObject,
    normalizeArray,
    normalizeText,
    normalizeStringArray,
    mergeStringArrays,
    resolveSkillData
} from "./candidateUtils.js";


import {
    buildCandidateExamples,
    mergeCandidateExamples
} from "./candidateExamples.js";


import {
    buildCandidateLearningMetrics
} from "./candidateMetrics.js";


import {
    resolveImprovementRequiredTools
} from "./candidateToolKnowledge.js";


/*
 * =========================================================
 * BUILD
 * =========================================================
 */


export function buildSkillImprovementCandidate({

    skills,

    trace,

    confidence,

    maturity,

    occurrences

}) {

    const skillList =

        Array.isArray(
            skills
        )

            ? skills.filter(
                isObject
            )

            : [];


    if(
        skillList.length === 0
    ){

        return null;

    }


    const rawExistingSkill =

        skillList[0];


    const existingSkill =

        resolveSkillData(
            rawExistingSkill
        );


    if(
        !existingSkill
    ){

        return null;

    }


    /*
     * =====================================================
     * EXAMPLES
     * =====================================================
     */


    const newExamples =

        buildCandidateExamples(
            trace
        );


    const examples =

        mergeCandidateExamples(

            normalizeArray(
                existingSkill.examples
            ),

            newExamples

        );


    /*
     * =====================================================
     * METRICS
     * =====================================================
     */


    const metrics =

        buildCandidateLearningMetrics({

            examples,

            occurrences,

            maturity,

            confidence

        });


    /*
     * =====================================================
     * TARGET
     * =====================================================
     */


    const targetSkillId =

        normalizeText(

            existingSkill.id

            ||

            existingSkill.skillId

            ||

            rawExistingSkill.id

            ||

            rawExistingSkill.skillId

        )

        ||

        null;


    /*
     * =====================================================
     * TOOL KNOWLEDGE
     * =====================================================
     */


    const toolKnowledge =

        resolveImprovementRequiredTools({

            existingSkill,

            trace

        });


    /*
     * =====================================================
     * RESULT
     * =====================================================
     */


    return {

        /*
         * Existing Skill transport.
         */


        skills:

            skillList,


        targetSkillId,


        baseVersion:

            Number(
                existingSkill.version || 0
            )

            ||

            null,


        /*
         * =================================================
         * IDENTITY
         * =================================================
         */


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


        /*
         * =================================================
         * KNOWLEDGE
         * =================================================
         */


        workflow:

            normalizeArray(
                existingSkill.workflow
            ),


        validationRules:

            normalizeStringArray(
                existingSkill.validationRules
            ),


        triggerPatterns:

            mergeStringArrays(

                existingSkill.triggerPatterns,

                existingSkill.keywords

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


        requiredTools:

            toolKnowledge.requiredTools,


        /*
         * =================================================
         * EVIDENCE
         * =================================================
         */


        examples,


        /*
         * =================================================
         * METRICS
         * =================================================
         */


        ...metrics,


        /*
         * =================================================
         * IMPROVEMENT META
         * =================================================
         */


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
