/*
 * =========================================================
 * JESSICA SKILL EVOLUTION SNAPSHOT
 * =========================================================
 */


import {
    fingerprintSkillKnowledge
} from "./skillKnowledgeFingerprint.js";


function normalizeText(
    value
) {

    return String(
        value || ""
    )
        .trim()
        .toUpperCase();

}


function countExamples(
    skill
) {

    return Array.isArray(
        skill?.examples
    )

        ? skill.examples.length

        : 0;

}


export function buildSkillEvolutionSnapshot({

    existingSkill,

    proposedKnowledge,

    proposedMaturityLevel

}) {

    const baseKnowledgeHash =

        fingerprintSkillKnowledge(
            existingSkill
        );


    const proposedKnowledgeHash =

        fingerprintSkillKnowledge(
            proposedKnowledge
        );


    const baseMaturityLevel =

        normalizeText(

            existingSkill
                ?.learning
                ?.maturityLevel

            ||

            existingSkill
                ?.maturityLevel

        );


    const nextMaturityLevel =

        normalizeText(
            proposedMaturityLevel
        );


    const baseExamplesCount =

        countExamples(
            existingSkill
        );


    const proposedExamplesCount =

        countExamples(
            proposedKnowledge
        );


    return {

        baseKnowledgeHash,

        proposedKnowledgeHash,

        knowledgeChanged:

            baseKnowledgeHash
            !==
            proposedKnowledgeHash,

        baseMaturityLevel:

            baseMaturityLevel

            ||

            null,

        proposedMaturityLevel:

            nextMaturityLevel

            ||

            null,

        maturityChanged:

            Boolean(
                baseMaturityLevel
                &&
                nextMaturityLevel
                &&
                baseMaturityLevel !==
                nextMaturityLevel
            ),

        baseExamplesCount,

        proposedExamplesCount,

        newEvidenceCount:

            Math.max(

                0,

                proposedExamplesCount
                -
                baseExamplesCount

            )

    };

}
