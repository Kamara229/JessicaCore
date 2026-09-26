/*
 * =========================================================
 * JESSICA LEARNING QUEUE
 * =========================================================
 *
 * Очередь новых знаний Jessica.
 *
 *
 * Flow:
 *
 * Learning Event
 *        ↓
 * Queue Item
 *        ↓
 * PENDING
 *        ↓
 * Approval
 *
 *
 * НЕ:
 *
 * - сохраняет Skill;
 * - изменяет Experience;
 * - пишет в Supabase.
 *
 * Только управляет состоянием обучения.
 *
 * =========================================================
 */



/*
 * =========================================================
 * CREATE QUEUE ITEM
 * =========================================================
 */


export function createLearningQueueItem(
    learningEvent
) {


    if (
        !learningEvent ||
        typeof learningEvent !== "object"
    ) {

        return null;

    }



    return {


        id:
            createQueueId(),



        status:
            "PENDING",



        action:
            learningEvent.action ||
            "IGNORE",



        skillId:
            learningEvent.skillId ||
            null,



        confidence:
            learningEvent.confidence ||
            0,



        event:
            learningEvent,



        createdAt:
            new Date()
                .toISOString(),



        reviewedAt:
            null


    };


}





/*
 * =========================================================
 * UPDATE STATUS
 * =========================================================
 */


export function updateLearningQueueStatus(
    
    item,

    status

) {


    if (
        !item
    ) {

        return null;

    }



    item.status =
        status;



    if (
        status === "APPROVED" ||
        status === "REJECTED"
    ) {

        item.reviewedAt =
            new Date()
                .toISOString();

    }



    return item;

}





/*
 * =========================================================
 * VALID ACTIONS
 * =========================================================
 */


export function isValidLearningAction(
    action
) {


    return [

        "NEW_SKILL",

        "SKILL_IMPROVEMENT",

        "IGNORE"

    ]
    .includes(
        action
    );

}





/*
 * =========================================================
 * ID
 * =========================================================
 */


function createQueueId() {


    try {


        if (
            typeof crypto !== "undefined" &&
            crypto.randomUUID
        ) {

            return crypto.randomUUID();

        }


    } catch(error) {

    }



    return (

        Date.now()
        +
        "-"
        +
        Math.random()
            .toString(36)
            .substring(2)

    );


}
