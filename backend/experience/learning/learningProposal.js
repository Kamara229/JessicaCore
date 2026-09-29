import {
    randomUUID
} from "node:crypto";



/*
 * =========================================================
 * JESSICA LEARNING PROPOSAL v3
 * =========================================================
 *
 * Промежуточная модель кандидата обучения.
 *
 *
 * Flow:
 *
 * Execution Trace
 *        ↓
 * Experience Analyzer
 *        ↓
 * Learning Router
 *        ↓
 * Learning Proposal
 *        ↓
 * Autonomy Policy
 *        ↓
 * Experience Skill
 *
 *
 * Proposal != Skill
 *
 *
 * НЕ:
 *
 * - сохраняет Skill;
 * - работает с БД;
 * - принимает решение обучения.
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

    if(
        !Array.isArray(value)
    ){

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
 * CONFIDENCE
 * =========================================================
 */


function resolveConfidence(
    queueItem,
    candidate
) {


    const queueConfidence =
        Number(
            queueItem?.confidence
        );



    if(
        Number.isFinite(
            queueConfidence
        )
    ){

        return queueConfidence;

    }



    const candidateConfidence =
        Number(
            candidate?.confidence
        );



    if(
        Number.isFinite(
            candidateConfidence
        )
    ){

        return candidateConfidence;

    }



    return 0;

}








/*
 * =========================================================
 * TARGET SKILL
 * =========================================================
 */


function resolveTargetSkill(
    queueItem,
    event
) {


    const candidate =

        event
            ?.payload
            ?.skillCandidate;



    /*
     * NEW SKILL
     */


    if(
        candidate?.skillId
    ){

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
     * EXISTING SKILL
     */


    const skills =

        event
            ?.payload
            ?.skills;



    if(
        Array.isArray(skills)
        &&
        skills.length
    ){

        const skill =
            skills[0];


        return {

            id:

                skill.id ||

                skill.skillId ||

                queueItem.skillId ||

                null,


            version:

                skill.version || null,


            exists:
                true

        };

    }







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
 * BUILD EXPERIENCE
 * =========================================================
 */


function buildProposedExperience(
    candidate,
    targetSkill
) {


    return {


        id:

            targetSkill.id || null,



        name:

            candidate.name ||

            `Jessica Skill ${targetSkill.id || "generated"}`,



        description:

            candidate.description || "",



        category:

            candidate.category ||

            "general",



        workflow:

            Array.isArray(
                candidate.workflow
            )

                ? candidate.workflow

                : [],






        validationRules:

            normalizeArray(
                candidate.validationRules
            ),






        triggerPatterns:

            normalizeArray(
                candidate.triggerPatterns
            ),





        examples:

            Array.isArray(
                candidate.examples
            )

                ? candidate.examples

                : [],






        constraints:

            normalizeArray(
                candidate.constraints
            ),





        /*
         * Новые поля автономного обучения
         */


        maturity:

            Number(
                candidate.maturity || 0
            ),



        occurrences:

            Number(
                candidate.occurrences || 0
            ),



        successRate:

            Number(
                candidate.successRate || 0
            ),




        failurePatterns:

            normalizeArray(
                candidate.failurePatterns
            ),




        avoidPatterns:

            normalizeArray(
                candidate.avoidPatterns
            ),





        requiredTools:

            normalizeArray(
                candidate.requiredTools
            ),





        source:

            candidate.source ||

            "execution-learning"


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


    if(
        !queueItem ||
        typeof queueItem !== "object"
    ){

        throw new Error(
            "Learning Proposal: queue item отсутствует"
        );

    }






    const event =
        extractEvent(
            queueItem
        );






    const candidate =

        event
            ?.payload
            ?.skillCandidate ||

        {};






    const targetSkill =

        resolveTargetSkill(
            queueItem,
            event
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

            "IGNORE",






        confidence:

            resolveConfidence(
                queueItem,
                candidate
            ),







        analysis:

        {


            reusable:

                event.reusable === true
                ||
                event.payload?.reusable === true,



            reason:

                event.reason || "",



            confidence:

                resolveConfidence(
                    queueItem,
                    candidate
                ),



            source:

                "experience-analyzer"

        },








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

            buildProposedExperience(

                candidate,

                targetSkill

            ),








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

    proposedExperience = null

} = {}) {


    const cleanTask =
        normalizeText(
            task
        );



    if(
        !cleanTask
    ){

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




        confidence:

            0,




        analysis:

        {

            reusable:
                false,


            reason:
                "manual",


            source:
                "manual"

        },




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


    if(
        !proposal ||
        typeof proposal !== "object"
    ){

        throw new Error(
            "Learning Proposal отсутствует"
        );

    }



    if(
        proposal.status !==
        LEARNING_PROPOSAL_STATUS.PENDING_APPROVAL
    ){

        throw new Error(
            "Learning Proposal уже обработан"
        );

    }


        }
