/*
 * =========================================================
 * JESSICA LEARNING REVIEWER v3
 * =========================================================
 *
 * Structural Validator
 * Learning Proposal.
 *
 *
 * Flow:
 *
 * Proposal
 *      ↓
 * Reviewer
 *      ↓
 * Quality Gate
 *
 *
 * Reviewer отвечает только за:
 *
 * - корректность Proposal;
 * - обязательные поля Experience;
 * - согласованность Action / Target Skill.
 *
 *
 * НЕ:
 *
 * - оценивает confidence;
 * - принимает AUTO_APPROVE;
 * - сохраняет Skill;
 * - работает с Supabase.
 *
 * =========================================================
 */


export const REVIEW_STATUS = {

    VALID:
        "VALID",

    INVALID:
        "INVALID"

};


const VALID_ACTIONS = [

    "NEW_SKILL",

    "SKILL_IMPROVEMENT"

];


function isObject(
    value
) {

    return (

        value &&
        typeof value === "object" &&
        !Array.isArray(value)

    );

}


function normalizeText(
    value
) {

    return String(
        value || ""
    )
    .trim();

}


function validateExperience(
    experience
) {

    const errors = [];


    if(
        !normalizeText(
            experience?.id
        )
    ){

        errors.push(
            "Experience ID отсутствует"
        );

    }


    if(
        !normalizeText(
            experience?.name
        )
    ){

        errors.push(
            "Experience name отсутствует"
        );

    }


    if(
        !Array.isArray(
            experience?.workflow
        )
        ||
        experience.workflow.length === 0
    ){

        errors.push(
            "Workflow отсутствует"
        );

    }


    if(
        !Array.isArray(
            experience?.examples
        )
        ||
        experience.examples.length === 0
    ){

        errors.push(
            "Examples отсутствуют"
        );

    }


    if(
        !isObject(
            experience?.learning
        )
    ){

        errors.push(
            "Learning metrics отсутствуют"
        );

    }


    return errors;

}


export function reviewLearningProposal(
    proposal
) {

    if(
        !isObject(
            proposal
        )
    ){

        return {

            status:
                REVIEW_STATUS.INVALID,

            valid:
                false,

            reason:
                "Proposal отсутствует"

        };

    }


    const action =

        normalizeText(
            proposal.action
        )
        .toUpperCase();


    if(
        !VALID_ACTIONS.includes(
            action
        )
    ){

        return {

            status:
                REVIEW_STATUS.INVALID,

            valid:
                false,

            reason:
                "Неподдерживаемый тип Learning Proposal"

        };

    }


    if(
        !isObject(
            proposal.proposedExperience
        )
    ){

        return {

            status:
                REVIEW_STATUS.INVALID,

            valid:
                false,

            reason:
                "Нет proposedExperience"

        };

    }


    const errors =

        validateExperience(
            proposal.proposedExperience
        );


    if(
        action === "SKILL_IMPROVEMENT"
    ){

        if(
            proposal?.targetSkill?.exists !== true
        ){

            errors.push(
                "Existing Skill не определён для Improvement"
            );

        }


        if(
            !normalizeText(
                proposal?.targetSkill?.id
            )
        ){

            errors.push(
                "Target Skill ID отсутствует"
            );

        }

    }


    if(
        errors.length > 0
    ){

        return {

            status:
                REVIEW_STATUS.INVALID,

            valid:
                false,

            errors,

            reason:
                errors.join("; ")

        };

    }


    return {

        status:
            REVIEW_STATUS.VALID,

        valid:
            true,

        proposalId:
            proposal.id || null,

        action,

        reason:
            "Learning Proposal структура корректна"

    };

}
