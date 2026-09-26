/*
 * =========================================================
 * JESSICA LEARNING QUEUE
 * =========================================================
 *
 * Очередь событий обучения Jessica.
 *
 *
 * Flow:
 *
 * Learning Event
 *        ↓
 * Queue Item
 *        ↓
 * Learning Worker
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
 * Только управляет очередью обучения.
 *
 * =========================================================
 */





/*
 * =========================================================
 * VALID ACTIONS
 * =========================================================
 */


const VALID_ACTIONS = [

    "NEW_SKILL",

    "SKILL_IMPROVEMENT",

    "IGNORE"

];






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





    const action =
        VALID_ACTIONS.includes(
            learningEvent.action
        )
            ? learningEvent.action
            : "IGNORE";






    /*
     * Получаем Skill ID
     *
     * NEW_SKILL:
     * payload.skillCandidate.skillId
     *
     * SKILL_IMPROVEMENT:
     * payload.skills[0].id
     *
     */


    const skillId =
        resolveSkillId(
            learningEvent
        );







    return {


        id:
            createQueueId(),



        status:

            "PENDING",



        action,



        skillId,



        confidence:

            Number(
                learningEvent.confidence || 0
            ),



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
 * RESOLVE SKILL ID
 * =========================================================
 */


function resolveSkillId(
    event
) {


    /*
     * Новый Skill
     */


    const candidate =
        event
            ?.payload
            ?.skillCandidate;



    if (
        candidate?.skillId
    ) {

        return candidate.skillId;

    }







    /*
     * Улучшение Skill
     */


    const skills =
        event
            ?.payload
            ?.skills;



    if (
        Array.isArray(skills) &&
        skills.length > 0
    ) {


        return (

            skills[0]?.id ||

            skills[0]?.skillId ||

            null

        );


    }





    return null;


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
 * VALID ACTION
 * =========================================================
 */


export function isValidLearningAction(
    action
) {


    return VALID_ACTIONS.includes(
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
