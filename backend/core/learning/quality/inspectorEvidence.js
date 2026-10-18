/*
 * =========================================================
 * JESSICA SKILL QUALITY — EVIDENCE
 * =========================================================
 */

import {
    resolveLearningExampleSuccess
} from "../exampleSuccess.js";


export function inspectSkillEvidence(
    experience,
    proposal = null
) {

    const errors = [];
    const warnings = [];

    const examples =

        Array.isArray(
            experience?.examples
        )

            ? experience.examples

            : [];


    let positiveExamples = 0;
    let contradictoryExamples = 0;


    for(
        const example
        of examples
    ){

        const canonicalSuccess =

            resolveLearningExampleSuccess(
                example
            );


        if(canonicalSuccess){

            positiveExamples++;

        }


        if(
            example?.success === true
            &&
            canonicalSuccess !== true
        ){

            contradictoryExamples++;

        }

    }


    // An older good example cannot justify an invalid new observation.
    if (proposal?.action === "SKILL_IMPROVEMENT") {
        const count = Number(proposal?.analysis?.evolution?.newEvidenceCount);
        if (!Number.isInteger(count) || count < 1 || count > examples.length) {
            errors.push("Improvement не содержит нового подтверждённого Evidence");
        } else if (examples.slice(-count).some(item => !resolveLearningExampleSuccess(item))) {
            errors.push("Новое Evidence не подтверждено: reinforcement запрещён");
        }
    }

    if(
        contradictoryExamples > 0
    ){

        errors.push(
            "Найдены Examples с success=true, которые каноническая проверка считает неуспешными"
        );

    }


    if(
        positiveExamples < 1
    ){

        errors.push(
            "Нет канонически подтверждённого положительного Example"
        );

    }


    if(
        examples.length > 0
        &&
        positiveExamples === examples.length
    ){

        warnings.push(
            "Skill содержит только положительные Examples; failure evidence пока отсутствует"
        );

    }


    return {

        errors,

        warnings,

        stats: {

            examples:
                examples.length,

            positiveExamples,

            contradictoryExamples

        }

    };

}
