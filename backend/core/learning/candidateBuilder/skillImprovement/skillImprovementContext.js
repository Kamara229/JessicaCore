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

import {verifiedDateExamples} from "./verifiedDateExamples.js";
import {
  calculateExperienceConfidence,calculateExperienceMaturity,calculateSuccessRate
} from "../../experienceConfidence.js";


import {
    resolveImprovementRequiredTools
} from "../candidateToolKnowledge.js";


import {
    resolveSkillExecutionProgram
} from "../../../../experience/execution/executionProgram.js";


import {
    buildSkillEvolutionSnapshot
} from "../../quality/skillEvolutionSnapshot.js";


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


    // A historic, incomplete response must not inflate future autonomous
    // reinforcement. Preserve immutable history but omit invalid examples.
    const audited = verifiedDateExamples(
        normalizeArray(existingSkill.examples),
        resolveSkillExecutionProgram(existingSkill)
    );

    const examples = mergeCandidateExamples(audited.examples,newExamples);

    const correctedOccurrences = audited.removed
        ? Math.max(examples.length,1) : occurrences;
    const correctedMaturity = audited.removed
        ? calculateExperienceMaturity(correctedOccurrences) : maturity;
    const correctedConfidence = audited.removed
        ? calculateExperienceConfidence({
            matchScore:1,
            successRate:calculateSuccessRate(examples),
            occurrences:correctedOccurrences
          }) : confidence;

    const metrics = buildCandidateLearningMetrics({
        examples,
        occurrences:correctedOccurrences,
        maturity:correctedMaturity,
        confidence:correctedConfidence
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


    const executionProgram =

        resolveSkillExecutionProgram(
            existingSkill
        );


    const proposedKnowledge = {

        ...existingSkill,

        requiredTools:
            toolKnowledge.requiredTools,

        executionProgram,

        examples

    };


    const evolution =

        buildSkillEvolutionSnapshot({

            existingSkill,

            proposedKnowledge,

            proposedMaturityLevel:
                metrics.maturityLevel

        });


    return {

        rawExistingSkill,

        existingSkill,

        examples,

        metrics,

        targetSkillId,

        toolKnowledge,

        executionProgram,

        evolution

    };

}
