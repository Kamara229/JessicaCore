/*
 * =========================================================
 * JESSICA LEARNING REVIEWER
 * =========================================================
 *
 * Проверка Learning Proposal перед Approval.
 *
 *
 * Flow:
 *
 * Learning Proposal
 *        ↓
 * Reviewer
 *        ↓
 * Review Decision
 *
 *
 * Возможные решения:
 *
 * APPROVE_READY
 *      ↓
 * можно передавать дальше
 *
 *
 * NEEDS_CLARIFICATION
 *      ↓
 * данных недостаточно
 *
 *
 * REJECT
 *      ↓
 * обучение не имеет ценности
 *
 *
 * НЕ:
 *
 * - создаёт Skill;
 * - сохраняет Experience;
 * - изменяет Supabase.
 *
 * =========================================================
 */





export const REVIEW_STATUS = {


    APPROVE_READY:
        "APPROVE_READY",


    NEEDS_CLARIFICATION:
        "NEEDS_CLARIFICATION",


    REJECT:
        "REJECT"


};





/*
 * =========================================================
 * VALIDATE PROPOSAL
 * =========================================================
 */


function isValidProposal(
    proposal
) {


    return (

        proposal &&

        typeof proposal === "object" &&

        proposal.proposedExperience

    );

}





/*
 * =========================================================
 * REVIEW EXPERIENCE
 * =========================================================
 */


function reviewExperience(
    experience
) {


    const problems =
        [];



    if (
        !experience.name
    ) {

        problems.push(
            "Отсутствует название навыка"
        );

    }



    if (
        !experience.workflow ||
        experience.workflow.length === 0
    ) {

        problems.push(
            "Отсутствует описание процесса"
        );

    }



    if (
        !experience.examples ||
        experience.examples.length === 0
    ) {

        problems.push(
            "Нет примеров использования"
        );

    }



    return problems;

}





/*
 * =========================================================
 * REVIEW PROPOSAL
 * =========================================================
 */


export function reviewLearningProposal(
    proposal
) {


    if (
        !isValidProposal(
            proposal
        )
    ) {


        return {


            status:
                REVIEW_STATUS.REJECT,


            approved:
                false,


            reason:
                "Некорректный Learning Proposal"


        };

    }





    const problems =
        reviewExperience(
            proposal.proposedExperience
        );





    if (
        problems.length > 0
    ) {


        return {


            status:
                REVIEW_STATUS
                    .NEEDS_CLARIFICATION,


            approved:
                false,


            reason:
                problems.join(
                    "; "
                ),


            problems



        };

    }





    return {


        status:
            REVIEW_STATUS
                .APPROVE_READY,


        approved:
            true,


        confidence:
            proposal.confidence || 0,


        reason:
            "Proposal готов к Approval"


    };


}
