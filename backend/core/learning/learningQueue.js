/*
 * =========================================================
 * JESSICA LEARNING QUEUE v2
 * =========================================================
 *
 * Создаёт Queue Item
 * из Learning Event.
 *
 *
 * Actions:
 *
 * NEW_SKILL
 * SKILL_IMPROVEMENT
 * PATTERN_DISCOVERY
 *
 *
 * IGNORE не сохраняется
 * Learning Coordinator'ом.
 *
 * =========================================================
 */


const VALID_ACTIONS = [

    "NEW_SKILL",

    "SKILL_IMPROVEMENT",

    "PATTERN_DISCOVERY",

    "IGNORE"

];





export function createLearningQueueItem(
    learningEvent
) {


    if(
        !learningEvent ||
        typeof learningEvent !== "object"
    ){

        return null;

    }



    const action =

        VALID_ACTIONS.includes(
            learningEvent.action
        )

            ? learningEvent.action

            : "IGNORE";



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





function resolveSkillId(
    event
) {


    /*
     * NEW SKILL
     */


    const candidate =

        event
            ?.payload
            ?.skillCandidate;



    if(
        candidate?.skillId
    ){

        return candidate.skillId;

    }



    /*
     * SKILL IMPROVEMENT
     */


    const skills =

        event
            ?.payload
            ?.skills;



    if(
        Array.isArray(skills)
        &&
        skills.length > 0
    ){

        return (

            skills[0]?.id

            ||

            skills[0]?.skillId

            ||

            null

        );

    }



    /*
     * PATTERN_DISCOVERY
     *
     * На этом этапе Skill ещё
     * не существует.
     */


    return null;

}





export function updateLearningQueueStatus(

    item,

    status

) {


    if(
        !item
    ){

        return null;

    }



    item.status =
        status;



    if(
        status === "APPROVED"
        ||
        status === "REJECTED"
        ||
        status === "IGNORED"
        ||
        status === "FAILED"
    ){

        item.reviewedAt =

            new Date()
                .toISOString();

    }



    return item;

}





export function isValidLearningAction(
    action
) {


    return VALID_ACTIONS.includes(
        action
    );

}





function createQueueId()
{

    try {


        if(
            typeof crypto !== "undefined"
            &&
            crypto.randomUUID
        ){

            return crypto.randomUUID();

        }


    }catch(error){

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
