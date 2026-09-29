/*
 * =========================================================
 * JESSICA LEARNING REVIEWER v2
 * =========================================================
 *
 * Технический валидатор Learning Proposal.
 *
 *
 * НЕ принимает решение обучения.
 *
 *
 * Flow:
 *
 * Learning Proposal
 *        ↓
 * Reviewer
 *        ↓
 * Validation Result
 *        ↓
 * Approval Runner
 *
 *
 * Решение:
 *
 * Learning Autonomy Policy
 *
 *
 * НЕ:
 *
 * - создаёт Skill;
 * - сохраняет Experience;
 * - меняет память.
 *
 * =========================================================
 */






export const REVIEW_STATUS = {


    VALID:
        "VALID",


    INVALID:
        "INVALID"

};









/*
 * =========================================================
 * CHECK OBJECT
 * =========================================================
 */


function isObject(
    value
){

    return (

        value &&

        typeof value === "object"

    );

}









/*
 * =========================================================
 * VALIDATE EXPERIENCE
 * =========================================================
 */


function validateExperience(
    experience
){

    const errors = [];




    if(
        !experience.name ||
        typeof experience.name !== "string"
    ){

        errors.push(
            "Experience name отсутствует"
        );

    }





    if(
        !Array.isArray(
            experience.workflow
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
            experience.examples
        )
        ||
        experience.examples.length === 0
    ){

        errors.push(
            "Examples отсутствуют"
        );

    }






    return errors;

}









/*
 * =========================================================
 * REVIEW PROPOSAL
 * =========================================================
 */


export function reviewLearningProposal(
    proposal
){

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






    if(
        !proposal.proposedExperience ||
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



        action:
            proposal.action || null,



        confidence:
            Number(
                proposal.confidence || 0
            ),



        reason:
            "Learning Proposal структура корректна"



    };


}
