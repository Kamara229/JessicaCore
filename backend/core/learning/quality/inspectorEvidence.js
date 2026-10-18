/*
 * =========================================================
 * JESSICA SKILL QUALITY — EVIDENCE
 * =========================================================
 */

import {
    resolveLearningExampleSuccess
} from "../exampleSuccess.js";


export function inspectSkillEvidence(
    experience
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
