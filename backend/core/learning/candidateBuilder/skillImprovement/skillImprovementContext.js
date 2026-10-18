/*
 * =========================================================
 * JESSICA SKILL IMPROVEMENT CONTEXT
 * =========================================================
 */


import {
    isObject,
    normalizeArray,
    normalizeText,
    resolveSkillData
} from "../candidateUtils.js";


import {
    buildCandidateExamples,
    mergeCandidateExamples
} from "../candidateExamples.js";


import {
    buildCandidateLearningMetrics
} from "../candidateMetrics.js";


import {
    resolveImprovementRequiredTools
} from "../candidateToolKnowledge.js";


export function buildSkillImprovementContext({

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


    if(!existingSkill){

        return null;

    }


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


    const metrics =

        buildCandidateLearningMetrics({

            examples,

            occurrences,

            maturity,

            confidence

        });


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


    const toolKnowledge =

        resolveImprovementRequiredTools({

            existingSkill,

            examples

        });


    return {

        rawExistingSkill,

        existingSkill,

        examples,

        metrics,

        targetSkillId,

        toolKnowledge

    };

}
