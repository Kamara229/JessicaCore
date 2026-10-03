import {
    randomUUID
} from "node:crypto";


/*
 * =========================================================
 * JESSICA LEARNING PROPOSAL v4
 * =========================================================
 *
 * Каноническая промежуточная модель
 * кандидата автономного обучения Jessica.
 *
 *
 * Flow:
 *
 * Learning Queue
 *        ↓
 * Learning Candidate
 *        ↓
 * Learning Proposal
 *        ↓
 * Reviewer
 *        ↓
 * Quality / Autonomy
 *        ↓
 * Experience Skill
 *
 *
 * Источники Candidate:
 *
 * 1. Known Experience Pattern
 *        ↓
 * NEW_SKILL
 *
 * 2. AI Pattern Discovery
 *        ↓
 * NEW_SKILL
 *
 * 3. Existing Experience Usage
 *        ↓
 * SKILL_IMPROVEMENT
 *
 *
 * Proposal != Skill
 *
 *
 * Ответственность:
 *
 * - принять Queue Item;
 * - извлечь канонический Candidate;
 * - определить Learning Action;
 * - определить Target Skill;
 * - сохранить Learning Metrics;
 * - сформировать полный proposedExperience;
 * - сохранить provenance обучения.
 *
 *
 * НЕ:
 *
 * - сохраняет Skill;
 * - работает с Supabase;
 * - принимает AUTO_APPROVE;
 * - создаёт новую версию Skill;
 * - вызывает AI;
 * - анализирует Execution Trace;
 * - объединяет KEEP_CANDIDATE;
 * - принимает решение о качестве опыта.
 *
 * =========================================================
 */


/*
 * =========================================================
 * STATUSES
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
 * ACTIONS
 * =========================================================
 */


export const LEARNING_PROPOSAL_ACTION = {

    NEW_SKILL:
        "NEW_SKILL",

    SKILL_IMPROVEMENT:
        "SKILL_IMPROVEMENT"

};


const VALID_ACTIONS = [

    LEARNING_PROPOSAL_ACTION.NEW_SKILL,

    LEARNING_PROPOSAL_ACTION.SKILL_IMPROVEMENT

];


/*
 * =========================================================
 * NORMALIZE
 * =========================================================
 */


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


function normalizeNumber(
    value
) {

    const number =
        Number(value);


    return Number.isFinite(number)

        ? number

        : 0;

}


function normalizeUnit(
    value
) {

    return Math.max(

        0,

        Math.min(

            1,

            normalizeNumber(
                value
            )

        )

    );

}


function normalizePositiveInteger(
    value
) {

    const number =
        Math.floor(
            normalizeNumber(
                value
            )
        );


    return number > 0

        ? number

        : 0;

}


function normalizeStringArray(
    value
) {

    if(
        !Array.isArray(
            value
        )
    ){

        return [];

    }


    const result = [];


    for(
        const item
        of value
    ){

        const normalized =

            normalizeText(
                item
            );


        if(
            !normalized
        ){

            continue;

        }


        if(
            !result.includes(
                normalized
            )
        ){

            result.push(
                normalized
            );

        }

    }


    return result;

}


function normalizeObjectArray(
    value
) {

    if(
        !Array.isArray(
            value
        )
    ){

        return [];

    }


    return value

        .filter(
            isObject
        )

        .map(

            item => ({
                ...item
            })

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


function resolveAction(
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
            VALID_ACTIONS.includes(
                action
            )
        ){

            return action;

        }

    }


    /*
     * PATTERN_DISCOVERY не должен
     * доходить до Proposal напрямую.
     *
     * Pattern Discovery Worker обязан
     * сначала превратить его в NEW_SKILL.
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


function resolveCandidate(
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


    if(
        Array.isArray(
            candidate?.skills
        )
        &&
        candidate.skills.length > 0
    ){

        return candidate.skills.filter(
            isObject
        );

    }


    return [];

}


/*
 * =========================================================
 * SKILL DATA
 * =========================================================
 *
 * Experience Search / Memory может
 * передавать Skill напрямую:
 *
 * {
 *     id,
 *     name,
 *     ...
 * }
 *
 * или через wrapper:
 *
 * {
 *     experience: {
 *         id,
 *         name,
 *         ...
 *     }
 * }
 *
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


    if(
        isObject(
            value.experience
        )
    ){

        return value.experience;

    }


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
 * Candidate является самым близким
 * источником Learning Metrics.
 *
 * Поэтому приоритет:
 *
 * candidate
 *      ↓
 * event
 *      ↓
 * queue
 *
 * =========================================================
 */


function resolveConfidence(
    queueItem,
    event,
    candidate
) {

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
 * КРИТИЧНО:
 *
 * Target определяется Learning Action,
 * а НЕ просто наличием candidate.skillId.
 *
 *
 * NEW_SKILL:
 *
 * Candidate создаёт новую lineage.
 *
 *
 * SKILL_IMPROVEMENT:
 *
 * Existing Experience имеет приоритет,
 * даже если Candidate в будущем
 * также получит skillId.
 *
 * =========================================================
 */


function resolveTargetSkill({

    queueItem,

    event,

    candidate,

    action

}) {


    /*
     * =====================================================
     * SKILL IMPROVEMENT
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

            skills[0] ||

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


/*
 * =========================================================
 * VALIDATE CANDIDATE
 * =========================================================
 *
 * Здесь проверяется именно транспортный
 * контракт Candidate.
 *
 * Семантическое качество дальше
 * проверяет Learning Reviewer.
 *
 * Не создаём "Jessica Skill generated",
 * потому что такой fallback скрывал бы
 * повреждение Learning Pipeline.
 *
 * =========================================================
 */


function validateCandidateForProposal({

    candidate,

    targetSkill,

    action

}) {


    if(
        !isObject(
            candidate
        )
    ){

        throw new Error(
            "Learning Proposal: Skill Candidate отсутствует"
        );

    }


    if(
        !normalizeText(
            candidate.name
        )
    ){

        throw new Error(
            "Learning Proposal: Candidate name отсутствует"
        );

    }


    if(
        !Array.isArray(
            candidate.workflow
        )
        ||
        candidate.workflow.length === 0
    ){

        throw new Error(
            "Learning Proposal: Candidate workflow отсутствует"
        );

    }


    if(
        !Array.isArray(
            candidate.examples
        )
        ||
        candidate.examples.length === 0
    ){

        throw new Error(
            "Learning Proposal: Candidate examples отсутствуют"
        );

    }


    if(
        !targetSkill?.id
    ){

        throw new Error(
            "Learning Proposal: Target Skill ID отсутствует"
        );

    }


    if(
        action ===
        LEARNING_PROPOSAL_ACTION.SKILL_IMPROVEMENT
        &&
        targetSkill.exists !== true
    ){

        throw new Error(
            "Learning Proposal: Existing Skill не определён для Improvement"
        );

    }


    return true;

}


/*
 * =========================================================
 * LEARNING METRICS
 * =========================================================
 */


function buildLearningMetrics(
    candidate,
    confidence
) {

    const examples =

        normalizeObjectArray(
            candidate?.examples
        );


    const calculatedSuccessCount =

        examples.filter(

            item =>
                item?.success === true

        )
        .length;


    const calculatedFailureCount =

        examples.filter(

            item =>
                item?.success === false

        )
        .length;


    const successCount =

        normalizePositiveInteger(
            candidate?.successCount
        )

        ||

        calculatedSuccessCount;


    const failureCount =

        normalizePositiveInteger(
            candidate?.failureCount
        )

        ||

        calculatedFailureCount;


    const occurrences =

        Math.max(

            normalizePositiveInteger(
                candidate?.occurrences
            ),

            examples.length,

            1

        );


    let successRate;


    if(
        candidate?.successRate !== undefined
        &&
        candidate?.successRate !== null
    ){

        successRate =

            normalizeUnit(
                candidate.successRate
            );

    }else{


        successRate =

            examples.length > 0

                ?

                Number(
                    (
                        calculatedSuccessCount /
                        examples.length
                    )
                    .toFixed(2)
                )

                :

                0;

    }


    return {


        occurrences,


        successCount,


        failureCount,


        successRate,


        maturity:

            normalizeUnit(
                candidate?.maturity
            ),


        maturityLevel:

            normalizeText(
                candidate?.maturityLevel
            )

            ||

            null,


        confidence:

            normalizeUnit(
                confidence
            )

    };

}


/*
 * =========================================================
 * BUILD PROPOSED EXPERIENCE
 * =========================================================
 *
 * Proposed Experience является
 * полным состоянием будущего Skill.
 *
 *
 * Это важно для:
 *
 * NEW_SKILL
 *        ↓
 * v1
 *
 *
 * SKILL_IMPROVEMENT
 *        ↓
 * полноценный vNext
 *
 *
 * Skill Builder в дальнейшем
 * НЕ должен самостоятельно угадывать,
 * что нужно объединить.
 *
 * =========================================================
 */


function buildProposedExperience({

    candidate,

    targetSkill,

    confidence

}) {


    const learning =

        buildLearningMetrics(
            candidate,
            confidence
        );


    return {


        /*
         * =================================================
         * IDENTITY
         * =================================================
         */


        id:

            targetSkill.id,



        name:

            normalizeText(
                candidate.name
            ),



        description:

            normalizeText(
                candidate.description
            ),



        category:

            normalizeText(
                candidate.category
            )

            ||

            "general",



        /*
         * =================================================
         * KNOWLEDGE
         * =================================================
         */


        workflow:

            Array.isArray(
                candidate.workflow
            )

                ? [...candidate.workflow]

                : [],



        triggerPatterns:

            normalizeStringArray(
                candidate.triggerPatterns
            ),



        keywords:

            normalizeStringArray(
                candidate.keywords
            ),



        tags:

            normalizeStringArray(
                candidate.tags
            ),



        validationRules:

            normalizeStringArray(
                candidate.validationRules
            ),



        constraints:

            normalizeStringArray(
                candidate.constraints
            ),



        strategy:

            normalizeStringArray(
                candidate.strategy
            ),



        sourcePriority:

            normalizeStringArray(
                candidate.sourcePriority
            ),



        requiredTools:

            normalizeStringArray(
                candidate.requiredTools
            ),



        successfulPatterns:

            normalizeStringArray(
                candidate.successfulPatterns
            ),



        failurePatterns:

            normalizeStringArray(
                candidate.failurePatterns
            ),



        avoidPatterns:

            normalizeStringArray(
                candidate.avoidPatterns
            ),



        /*
         * =================================================
         * EVIDENCE
         * =================================================
         */


        examples:

            normalizeObjectArray(
                candidate.examples
            ),



        /*
         * =================================================
         * LEARNING METRICS
         * =================================================
         *
         * Верхнеуровневые поля сохраняются
         * для совместимости текущего
         * Autonomy Policy.
         *
         * learning является новым
         * каноническим контейнером.
         *
         * =================================================
         */


        occurrences:

            learning.occurrences,



        successCount:

            learning.successCount,



        failureCount:

            learning.failureCount,



        successRate:

            learning.successRate,



        maturity:

            learning.maturity,



        maturityLevel:

            learning.maturityLevel,



        confidence:

            learning.confidence,



        learning,



        /*
         * =================================================
         * CANDIDATE META
         * =================================================
         */


        candidateType:

            normalizeText(
                candidate.candidateType
            )

            ||

            null,



        improvementType:

            normalizeText(
                candidate.improvementType
            )

            ||

            null,



        baseVersion:

            normalizePositiveInteger(
                candidate.baseVersion
            )

            ||

            targetSkill.version

            ||

            null,



        source:

            normalizeText(
                candidate.source
            )

            ||

            "execution-learning"

    };

}


/*
 * =========================================================
 * BUILD ANALYSIS SUMMARY
 * =========================================================
 */


function buildAnalysis({

    event,

    candidate,

    confidence,

    action

}) {


    const originalAnalysis =

        isObject(
            event?.analysis
        )

            ? event.analysis

            : {};


    const payloadSource =

        normalizeText(
            event?.payload?.source
        );


    const discovery =
