/*
 * =========================================================
 * JESSICA LEARNING PROPOSAL RESOLVER
 * =========================================================
 *
 * Извлекает данные,
 * необходимые для Learning Proposal.
 *
 *
 * Queue Item
 *      ↓
 * Event
 *      ↓
 * Action
 *      ↓
 * Candidate
 *      ↓
 * Target Skill
 *
 *
 * НЕ:
 *
 * - создаёт Proposal;
 * - создаёт Experience;
 * - принимает Approval.
 *
 * =========================================================
 */


import {
    LEARNING_PROPOSAL_ACTION,
    VALID_PROPOSAL_ACTIONS
} from "./proposalConstants.js";


import {
    isObject,
    normalizeText,
    normalizeUnit,
    normalizePositiveInteger
} from "./proposalUtils.js";


/*
 * =========================================================
 * EVENT
 * =========================================================
 */


export function extractProposalEvent(
    queueItem
) {

    const event =

        queueItem?.event

        ??

        queueItem?.event_json

        ??

        {};


    return isObject(
        event
    )

        ? event

        : {};

}


/*
 * =========================================================
 * ACTION
 * =========================================================
 */


export function resolveProposalAction(
    queueItem,
    event
) {

    const candidates = [

        queueItem?.action,

        event?.action

    ];


    for(
        const value
        of candidates
    ){

        const action =

            normalizeText(
                value
            )
            .toUpperCase();


        if(
            VALID_PROPOSAL_ACTIONS.includes(
                action
            )
        ){

            return action;

        }

    }


    /*
     * PATTERN_DISCOVERY сюда
     * попадать уже не должен.
     *
     * Pattern Discovery Worker обязан
     * сначала превратить его
     * в NEW_SKILL.
     */


    throw new Error(

        "Learning Proposal: неподдерживаемый Learning Action"

    );

}


/*
 * =========================================================
 * CANDIDATE
 * =========================================================
 */


export function resolveProposalCandidate(
    event
) {

    const candidate =

        event
            ?.payload
            ?.skillCandidate;


    return isObject(
        candidate
    )

        ? candidate

        : null;

}


/*
 * =========================================================
 * EXISTING SKILLS
 * =========================================================
 */


function resolveExistingSkills(
    event,
    candidate
) {

    const eventSkills =

        event
            ?.payload
            ?.skills;


    if(
        Array.isArray(
            eventSkills
        )
        &&
        eventSkills.length > 0
    ){

        return eventSkills.filter(
            isObject
        );

    }


    const candidateSkills =

        candidate?.skills;


    if(
        Array.isArray(
            candidateSkills
        )
        &&
        candidateSkills.length > 0
    ){

        return candidateSkills.filter(
            isObject
        );

    }


    return [];

}


/*
 * =========================================================
 * SKILL DATA
 * =========================================================
 */


function resolveSkillData(
    value
) {

    if(
        !isObject(
            value
        )
    ){

        return null;

    }


    /*
     * Experience wrapper
     */


    if(
        isObject(
            value.experience
        )
    ){

        return value.experience;

    }


    /*
     * Skill wrapper
     */


    if(
        isObject(
            value.skill
        )
    ){

        return value.skill;

    }


    return value;

}


/*
 * =========================================================
 * CONFIDENCE
 * =========================================================
 *
 * Candidate является ближайшим
 * источником Learning Metrics.
 *
 * Priority:
 *
 * Candidate
 *      ↓
 * Event
 *      ↓
 * Queue Item
 *
 * =========================================================
 */


export function resolveProposalConfidence({

    queueItem,

    event,

    candidate

}) {

    const values = [

        candidate?.confidence,

        event?.confidence,

        queueItem?.confidence

    ];


    for(
        const value
        of values
    ){

        if(
            value === undefined
            ||
            value === null
        ){

            continue;

        }


        const number =
            Number(value);


        if(
            Number.isFinite(
                number
            )
        ){

            return normalizeUnit(
                number
            );

        }

    }


    return 0;

}


/*
 * =========================================================
 * TARGET SKILL
 * =========================================================
 *
 * Тип Target определяется ACTION,
 * а не наличием candidate.skillId.
 *
 * =========================================================
 */


export function resolveProposalTargetSkill({

    queueItem,

    event,

    candidate,

    action

}) {


    /*
     * =====================================================
     * EXISTING SKILL
     * =====================================================
     */


    if(
        action ===
        LEARNING_PROPOSAL_ACTION.SKILL_IMPROVEMENT
    ){

        const skills =

            resolveExistingSkills(
                event,
                candidate
            );


        const rawSkill =

            skills[0]

            ||

            null;


        const skill =

            resolveSkillData(
                rawSkill
            );


        const id =

            normalizeText(

                candidate?.targetSkillId

                ||

                skill?.id

                ||

                skill?.skillId

                ||

                rawSkill?.id

                ||

                rawSkill?.skillId

                ||

                queueItem?.skillId

                ||

                queueItem?.skill_id

                ||

                event?.skillId

            )

            ||

            null;


        const version =

            normalizePositiveInteger(

                candidate?.baseVersion

                ||

                skill?.version

                ||

                rawSkill?.version

            )

            ||

            null;


        return {

            id,

            version,

            exists:

                Boolean(
                    id
                )

        };

    }


    /*
     * =====================================================
     * NEW SKILL
     * =====================================================
     */


    const id =

        normalizeText(

            candidate?.skillId

            ||

            candidate?.id

            ||

            queueItem?.skillId

            ||

            queueItem?.skill_id

            ||

            event?.skillId

        )

        ||

        null;


    return {

        id,

        version:
            null,

        exists:
            false

    };

}
