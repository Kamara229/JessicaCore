/*
 * =========================================================
 * JESSICA LEARNING TRIGGER v5
 * =========================================================
 *
 * Точка входа Learning Pipeline.
 *
 *
 * Flow:
 *
 * Execution Trace
 *        ↓
 * shouldAnalyze()
 *        ↓
 * Experience Analyzer
 *        ↓
 * Learning Router
 *        ↓
 * Learning Event
 *
 *
 * Actions:
 *
 * NEW_SKILL
 * SKILL_IMPROVEMENT
 * PATTERN_DISCOVERY
 * IGNORE
 *
 * =========================================================
 */


import {
    analyzeExecutionTrace
} from "./experienceAnalyzer.js";


import {
    routeLearningEvent
} from "./learningRouter.js";


/*
 * =========================================================
 * NORMALIZE
 * =========================================================
 */


function isObject(
    value
) {

    return Boolean(

        value

        &&

        typeof value === "object"

        &&

        !Array.isArray(value)

    );

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


function normalizeText(
    value
) {

    return String(
        value || ""
    )
    .trim();

}


/*
 * =========================================================
 * SHOULD ANALYZE
 * =========================================================
 *
 * Канонический источник:
 *
 * trace.statistics.completed
 *
 *
 * Compatibility:
 *
 * trace.stats.completed
 *
 *
 * Дополнительно поддерживаем
 * успешный terminal Trace.
 *
 * =========================================================
 */


function shouldAnalyze(
    trace
) {

    if(
        !isObject(trace)
    ){

        return false;

    }


    /*
     * =====================================================
     * COMPLETED EXECUTIONS
     * =====================================================
     */


    const completed =

        normalizeNumber(

            trace?.statistics?.completed

            ??

            trace?.stats?.completed

        );


    if(
        completed > 0
    ){

        return true;

    }


    /*
     * =====================================================
     * TRACE STATUS
     * =====================================================
     */


    const traceStatus =

        normalizeText(
            trace?.status
        )
        .toUpperCase();


    if(
        traceStatus ===
        "COMPLETED"
    ){

        return true;

    }


    /*
     * =====================================================
     * RESULT STATUS
     * =====================================================
     */


    const resultStatus =

        normalizeText(
            trace?.result?.status
        )
        .toUpperCase();


    if(
        resultStatus ===
        "COMPLETED"
    ){

        return true;

    }


    /*
     * Explicit success fallback.
     *
     * Не считаем semantic terminal states
     * успешными Learning executions.
     */


    if(
        trace?.result?.success === true

        &&

        ![
            "NO_VERIFIED_RESULT",
            "NEEDS_CLARIFICATION",
            "FAILED"
        ]
        .includes(
            resultStatus
        )
    ){

        return true;

    }


    return false;

}


/*
 * =========================================================
 * NORMALIZE PAYLOAD
 * =========================================================
 */


function normalizeLearningPayload({

    decision,

    analysis

}) {

    const routerPayload =

        isObject(
            decision?.payload
        )

            ? decision.payload

            : {};


    const analysisCandidate =

        isObject(
            analysis?.skillCandidate
        )

            ? analysis.skillCandidate

            : null;


    const payload = {

        ...routerPayload

    };


    /*
     * Candidate fallback.
     */


    if(
        !isObject(
            payload.skillCandidate
        )
        &&
        analysisCandidate
    ){

        payload.skillCandidate =
            analysisCandidate;

    }


    /*
     * Existing Skills fallback.
     */


    if(
        !Array.isArray(
            payload.skills
        )
        &&
        Array.isArray(
            analysisCandidate?.skills
        )
    ){

        payload.skills =
            analysisCandidate.skills;

    }


    /*
     * Source.
     */


    if(
        !normalizeText(
            payload.source
        )
    ){

        payload.source =
            "experience-analyzer";

    }


    return payload;

}


/*
 * =========================================================
 * RESOLVE SKILL ID
 * =========================================================
 */


function resolveSkillId({

    decision,

    payload

}) {

    /*
     * NEW SKILL.
     */


    const candidate =

        isObject(
            payload?.skillCandidate
        )

            ? payload.skillCandidate

            : null;


    const candidateSkillId =

        normalizeText(

            candidate?.skillId

            ||

            candidate?.id

        );


    if(
        candidateSkillId
    ){

        return candidateSkillId;

    }


    /*
     * EXISTING SKILL.
     */


    const skills =

        Array.isArray(
            payload?.skills
        )

            ? payload.skills

            : [];


    if(
        skills.length > 0
    ){

        const existingSkillId =

            normalizeText(

                skills[0]?.id

                ||

                skills[0]?.skillId

            );


        if(
            existingSkillId
        ){

            return existingSkillId;

        }

    }


    return (

        normalizeText(
            decision?.skillId
        )

        ||

        null

    );

}


/*
 * =========================================================
 * RESOLVE CONFIDENCE
 * =========================================================
 */


function resolveConfidence({

    decision,

    analysis,

    payload

}) {

    const values = [

        decision?.confidence,

        payload
            ?.skillCandidate
            ?.confidence,

        analysis
            ?.skillCandidate
            ?.confidence

    ];


    for(
        const value
        of values
    ){

        const number =
            Number(value);


        if(
            Number.isFinite(number)
        ){

            return Math.max(

                0,

                Math.min(
                    1,
                    number
                )

            );

        }

    }


    return 0;

}


/*
 * =========================================================
 * BUILD EVENT
 * =========================================================
 */


function buildLearningEvent({

    trace,

    analysis,

    decision

}) {

    const payload =

        normalizeLearningPayload({

            decision,

            analysis

        });


    const skillId =

        resolveSkillId({

            decision,

            payload

        });


    const confidence =

        resolveConfidence({

            decision,

            analysis,

            payload

        });


    const reusable =

        decision?.reusable === true

        ||

        analysis?.reusable === true;


    const reason =

        normalizeText(
            decision?.reason
        )

        ||

        normalizeText(
            analysis?.reason
        )

        ||

        "";


    return {

        /*
         * Identity.
         */


        id:

            trace?.id

            ||

            null,


        traceId:

            trace?.id

            ||

            trace?.traceId

            ||

            null,


        /*
         * Decision.
         */


        action:

            decision?.action

            ||

            analysis?.action

            ||

            "IGNORE",


        skillId,


        confidence,


        reusable,


        reason,


        /*
         * Payload.
         */


        payload,


        /*
         * Diagnostic context.
         */


        analysis,


        decision,


        source:

            "learning-trigger",


        createdAt:

            new Date()
                .toISOString()

    };

}


/*
 * =========================================================
 * EMPTY
 * =========================================================
 */


function buildEmptyResult(
    reason
) {

    return {

        triggered:
            false,

        reason:

            normalizeText(reason)

            ||

            "Learning не запущен",

        analysis:
            null,

        decision:
            null,

        learningEvent:
            null,

        traceId:
            null

    };

}


/*
 * =========================================================
 * ERROR
 * =========================================================
 */


function buildErrorResult({

    trace,

    error

}) {

    return {

        triggered:
            false,

        reason:
            "Ошибка Learning Pipeline",

        error:

            normalizeText(
                error?.message
            )

            ||

            "unknown error",

        analysis:
            null,

        decision:
            null,

        learningEvent:
            null,

        traceId:

            trace?.id

            ||

            trace?.traceId

            ||

            null

    };

}


/*
 * =========================================================
 * LOG
 * =========================================================
 */


function logLearningEvent(
    learningEvent
) {

    console.log(

        "Jessica Learning Event:",

        JSON.stringify({

            action:

                learningEvent?.action

                ||

                "IGNORE",

            skillId:

                learningEvent?.skillId

                ||

                null,

            confidence:

                learningEvent?.confidence

                ||

                0,

            reusable:

                learningEvent?.reusable === true,

            hasCandidate:

                isObject(
                    learningEvent
                        ?.payload
                        ?.skillCandidate
                ),

            existingSkills:

                Array.isArray(
                    learningEvent
                        ?.payload
                        ?.skills
                )

                    ? learningEvent
                        .payload
                        .skills
                        .length

                    : 0

        })

    );

}


/*
 * =========================================================
 * RUN
 * =========================================================
 */


export function runLearningTrigger(
    trace
) {

    /*
     * 1. Eligibility.
     */


    if(
        !shouldAnalyze(
            trace
        )
    ){

        return buildEmptyResult(

            "Недостаточно данных для обучения"

        );

    }


    try {


        /*
         * 2. Analyze.
         */


        const analysis =

            analyzeExecutionTrace(
                trace
            );


        if(
            !isObject(
                analysis
            )
        ){

            return buildEmptyResult(

                "Experience Analyzer не вернул результат"

            );

        }


        /*
         * 3. Route.
         */


        const decision =

            routeLearningEvent({

                trace,

                analysis

            });


        if(
            !isObject(
                decision
            )
        ){

            return buildEmptyResult(

                "Learning Router не вернул решение"

            );

        }


        /*
         * 4. Event.
         */


        const learningEvent =

            buildLearningEvent({

                trace,

                analysis,

                decision

            });


        /*
         * 5. Log.
         */


        logLearningEvent(
            learningEvent
        );


        /*
         * 6. Result.
         */


        return {

            triggered:
                true,

            analysis,

            decision,

            learningEvent,

            traceId:

                trace?.id

                ||

                trace?.traceId

                ||

                null

        };


    }catch(error){


        console.error(

            "Jessica Learning Trigger error:",

            error

        );


        return buildErrorResult({

            trace,

            error

        });

    }

}
