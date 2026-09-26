import {
    randomUUID
} from "node:crypto";



/*
 * =========================================================
 * JESSICA LEARNING PROPOSAL
 * =========================================================
 *
 * Модель предложения изменения памяти Jessica.
 *
 *
 * Proposal НЕ является Skill.
 *
 * Это кандидат:
 *
 * NEW_SKILL
 *      создание нового навыка
 *
 * SKILL_IMPROVEMENT
 *      улучшение существующего навыка
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
 * - работает с Supabase;
 * - изменяет Experience.
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
 * EVENT
 * =========================================================
 */


function extractEvent(
    queueItem
) {


    return (

        queueItem?.event ||

        queueItem?.event_json ||

        {}

    );

}









/*
 * =========================================================
 * RESOLVE SKILL TARGET
 * =========================================================
 */


function resolveTargetSkill(
    queueItem,
    event
) {


    /*
     * NEW_SKILL
     */


    const candidate =
        event
            ?.payload
            ?.skillCandidate;



    if (
        candidate?.skillId
    ) {

        return {


            id:
                candidate.skillId,


            version:
                null,


            exists:
                false


        };

    }








    /*
     * SKILL IMPROVEMENT
     */


    const skills =
        event
            ?.payload
            ?.skills;



    if (
        Array.isArray(skills) &&
        skills.length > 0
    ) {


        const skill =
            skills[0];



        return {


            id:

                skill.id ||

                skill.skillId ||

                queueItem.skillId ||

                null,



            version:

                skill.version ||

                null,



            exists:

                true


        };


    }








    /*
     * fallback
     */


    return {


        id:

            queueItem.skillId ||

            event.skillId ||

            null,


        version:

            null,


        exists:

            false


    };


}









/*
 * =========================================================
 * CREATE FROM QUEUE
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
            "Learning Proposal: queue item отсутствует"
        );

    }





    const event =
        extractEvent(
            queueItem
        );





    const action =
        queueItem.action ||

        event.action ||

        "IGNORE";





    const targetSkill =
        resolveTargetSkill(
            queueItem,
            event
        );





    const sourceExperience =

        event
            ?.payload
            ?.skillCandidate ||

        {};






    const skillName =

        sourceExperience.name ||

        `Jessica Skill ${targetSkill.id || "generated"}`;








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



        action,



        confidence:

            Number(
                queueItem.confidence || 0
            ),




        targetSkill:

        {


            id:

                targetSkill.id,


            version:

                targetSkill.version,


            exists:

                targetSkill.exists


        },






        proposedExperience:

        {


            id:

                targetSkill.id || null,



            name:

                skillName,



            description:

                sourceExperience.description ||

                "",



            workflow:

                Array.isArray(
                    sourceExperience.workflow
                )

                    ? sourceExperience.workflow

                    : [],



            triggerPatterns:

                normalizeArray(
                    sourceExperience.triggerPatterns
                ),



            examples:

                Array.isArray(
                    sourceExperience.examples
                )

                    ? sourceExperience.examples

                    : [],



            constraints:

                normalizeArray(
                    sourceExperience.constraints
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
 * MANUAL PROPOSAL
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



        action:

            "NEW_SKILL",



        task:

            cleanTask,



        previousAnswer:

            normalizeText(
                previousAnswer
            ),



        correction:

            normalizeText(
                correction
            ),



        correctedAnswer,



        understanding,



        clarificationQuestions:

            normalizeArray(
                clarificationQuestions
            ),



        proposedExperience,



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
            "Learning Proposal отсутствует"
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
