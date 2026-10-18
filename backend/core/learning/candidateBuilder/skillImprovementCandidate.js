/*
 * =========================================================
 * JESSICA SKILL IMPROVEMENT CANDIDATE BUILDER v4
 * =========================================================
 *
 * Existing Experience
 *        +
 * New Execution Evidence
 *        ↓
 * SKILL_IMPROVEMENT Candidate
 *
 *
 * Improvement:
 *
 * - сохраняет существующий Knowledge;
 * - добавляет новый Example;
 * - пересчитывает Learning Metrics;
 * - может восстановить requiredTools
 *   старого Skill по нескольким
 *   независимым Execution.
 *
 * =========================================================
 */


import {
    isObject,
    normalizeArray,
    normalizeText,
    normalizeStringArray,
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


import {
    resolveSkillExecutionProgram
} from "../../../experience/execution/executionProgram.js";


/*
 * =========================================================
 * COMPACT SKILL REFERENCE
 * =========================================================
 *
 * Candidate не должен транспортировать
 * полный Experience Skill обратно в Queue.
 *
 * Для Proposal достаточно identity/version.
 * Полный Knowledge уже скопирован в поля
 * самого Candidate выше по flow.
 * =========================================================
 */


function buildSkillReference(
    rawSkill,
    resolvedSkill
) {

    const id =

        normalizeText(

            resolvedSkill?.id

            ||

            resolvedSkill?.skillId

            ||

            rawSkill?.id

            ||

            rawSkill?.skillId

        );


    if(
        !id
    ){

        return null;

    }


    const version =

        Number(

            resolvedSkill?.version

            ??

            rawSkill?.version

            ??

            0

        );


    return {

        id,

        version:

            Number.isFinite(version)
            &&
            version > 0

                ? version

                : null

    };

}


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
     * TARGET SKILL
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
     *
     * Здесь используются ВСЕ накопленные
     * examples существующего Skill
     * плюс текущий Execution.
     *
     *
     * Благодаря legacy fallback:
     *
     * result.executionMeta.usedTools
     *
     * можно использовать и старые Examples,
     * созданные до появления
     * example.executedTools.
     *
     * =====================================================
     */


    const toolKnowledge =

        resolveImprovementRequiredTools({

            existingSkill,

            examples

        });


    /*
     * =====================================================
     * RESULT
     * =====================================================
     */


    return {

        /*
         * EXISTING SKILL TRANSPORT
         */


        skills:

            [

                buildSkillReference(

                    rawExistingSkill,

                    existingSkill

                )

            ]
                .filter(
                    Boolean
                ),


        targetSkillId,


        baseVersion:

            Number(
                existingSkill.version || 0
            )

            ||

            null,


        /*
         * IDENTITY
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
         * KNOWLEDGE
         */


        workflow:

            normalizeArray(
                existingSkill.workflow
            ),


        validationRules:

            normalizeStringArray(
                existingSkill.validationRules
            ),


        /*
         * Trigger Patterns и Keywords — разные сигналы.
         *
         * Keywords могут быть широкими:
         * "дата", "версия", "release".
         *
         * Автономный Improvement не должен каждый раз
         * превращать такие общие keywords в triggers,
         * иначе Skill постепенно начинает матчить
         * слишком широкий класс задач.
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


        /*
         * EVIDENCE
         */


        examples,


        /*
         * LEARNING METRICS
         */


        ...metrics,


        /*
         * IMPROVEMENT META
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
