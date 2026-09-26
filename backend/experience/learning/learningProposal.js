import {
    randomUUID
} from "node:crypto";


/*
 * =========================================================
 * JESSICA LEARNING PROPOSAL
 * =========================================================
 *
 * Модель предложения обучения Jessica.
 *
 *
 * Proposal — это НЕ Skill.
 *
 * Это кандидат на изменение памяти Jessica,
 * который проходит Approval.
 *
 *
 * Flow:
 *
 * Learning Queue
 *        ↓
 * Proposal
 *        ↓
 * Approval
 *        ↓
 * Experience Skill
 *
 *
 * НЕ:
 *
 * - сохраняет Skill;
 * - пишет в Supabase;
 * - изменяет Experience;
 * - принимает решение Approval.
 *
 * =========================================================
 */



export const LEARNING_PROPOSAL_STATUS = {


    PENDING_APPROVAL:
        "PENDING_APPROVAL",


    APPROVED:
        "APPROVED",


    REJECTED:
        "REJECTED"


};





/*
 * =========================================================
 * NORMALIZE
 * =========================================================
 */


function normalizeText(
    value
) {

    return String(
        value || ""
    )
        .trim();

}





function normalizeArray(
    value
) {

    if (
        !Array.isArray(value)
    ) {

        return [];

    }


    return value

        .map(
            item =>
                normalizeText(item)
        )

        .filter(
            Boolean
        );

}





/*
 * =========================================================
 * EXTRACT QUEUE EVENT
 * =========================================================
 */


function extractEvent(
    queueItem
) {

    return (

        queueItem?.event_json ||

        queueItem?.event ||

        {}

    );

}





/*
 * =========================================================
 * CREATE FROM QUEUE
 * =========================================================
 *
 * Основной путь автоматического обучения.
 *
 * =========================================================
 */


export function createLearningProposalFromQueue(
    queueItem
) {


    if (
        !queueItem ||
        typeof queueItem !== "object"
    ) {

        throw new Error(
            "Learning Proposal: queue item не указан"
        );

    }



    const event =
        extractEvent(
            queueItem
        );



    const proposedExperience =
        event.proposedExperience ||

        event.experience ||

        {};




    const skillName =

        proposedExperience.name ||

        event.skillName ||

        "Jessica Generated Skill";





    const skillId =

        proposedExperience.id ||

        event.skillId ||

        skillName

            .toLowerCase()

            .replace(
                /\s+/g,
                "_"
            );





    return {


        id:
            randomUUID(),



        status:
            LEARNING_PROPOSAL_STATUS
                .PENDING_APPROVAL,



        source:
            "learning_queue",



        queueItemId:
            queueItem.id || null,



        action:
            queueItem.action ||
            event.action ||
            "NEW_SKILL",



        confidence:
            Number(
                queueItem.confidence ||
                event.confidence ||
                0
            ),



        proposedExperience: {


            id:
                skillId,


            name:
                skillName,


            description:
                proposedExperience.description ||
                "",


            workflow:
                Array.isArray(
                    proposedExperience.workflow
                )
                    ? proposedExperience.workflow
                    : [],



            triggerPatterns:
                normalizeArray(
                    proposedExperience.triggerPatterns
                ),



            examples:
                Array.isArray(
                    proposedExperience.examples
                )
                    ? proposedExperience.examples
                    : [],



            constraints:
                normalizeArray(
                    proposedExperience.constraints
                )


        },



        createdAt:
            new Date()
                .toISOString(),



        approvedAt:
            null,


        rejectedAt:
            null


    };

}





/*
 * =========================================================
 * CREATE MANUAL PROPOSAL
 * =========================================================
 *
 * Для обучения через исправление пользователя.
 *
 * =========================================================
 */


export function createLearningProposal({

    task,

    previousAnswer = "",

    correction = "",

    correctedAnswer = "",

    understanding = "",

    clarificationQuestions = [],

    proposedExperience = null

} = {}) {


    const cleanTask =
        normalizeText(
            task
        );


    const cleanCorrection =
        normalizeText(
            correction
        );



    if (
        !cleanTask
    ) {

        throw new Error(
            "Learning Proposal: задача не указана"
        );

    }



    return {


        id:
            randomUUID(),



        status:
            LEARNING_PROPOSAL_STATUS
                .PENDING_APPROVAL,



        source:
            "manual",



        task:
            cleanTask,



        previousAnswer:
            normalizeText(
                previousAnswer
            ),



        correction:
            cleanCorrection,



        correctedAnswer:
            normalizeText(
                correctedAnswer
            ),



        understanding:
            normalizeText(
                understanding
            ),



        clarificationQuestions:
            normalizeArray(
                clarificationQuestions
            ),



        proposedExperience:
            proposedExperience || null,



        createdAt:
            new Date()
                .toISOString(),



        approvedAt:
            null,


        rejectedAt:
            null


    };

}





/*
 * =========================================================
 * APPROVE
 * =========================================================
 */


export function approveLearningProposal(
    proposal
) {


    validateProposal(
        proposal
    );


    return {

        ...proposal,


        status:
            LEARNING_PROPOSAL_STATUS
                .APPROVED,


        approvedAt:
            new Date()
                .toISOString(),


        rejectedAt:
            null

    };

}





/*
 * =========================================================
 * REJECT
 * =========================================================
 */


export function rejectLearningProposal(
    proposal
) {


    validateProposal(
        proposal
    );


    return {

        ...proposal,


        status:
            LEARNING_PROPOSAL_STATUS
                .REJECTED,


        approvedAt:
            null,


        rejectedAt:
            new Date()
                .toISOString()

    };

}





/*
 * =========================================================
 * VALIDATE
 * =========================================================
 */


function validateProposal(
    proposal
) {


    if (
        !proposal ||
        typeof proposal !== "object"
    ) {

        throw new Error(
            "Learning Proposal не указан"
        );

    }



    if (
        proposal.status !==
        LEARNING_PROPOSAL_STATUS.PENDING_APPROVAL
    ) {

        throw new Error(
            "Learning Proposal уже обработан"
        );

    }


        }
