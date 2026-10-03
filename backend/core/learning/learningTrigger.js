/*
 * =========================================================
 * JESSICA LEARNING TRIGGER v4
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
 * Возможные решения:
 *
 * NEW_SKILL
 *      ↓
 * кандидат нового Experience Skill
 *
 *
 * SKILL_IMPROVEMENT
 *      ↓
 * кандидат новой версии
 * существующего Experience Skill
 *
 *
 * IGNORE
 *      ↓
 * опыт не передаётся в Learning Queue
 *
 *
 * Ответственность:
 *
 * - принять Execution Trace;
 * - запустить Experience Analyzer;
 * - передать Analysis в Learning Router;
 * - сформировать полный Learning Event;
 * - сохранить Learning Payload без потери данных;
 * - определить Skill ID для следующих слоёв.
 *
 *
 * НЕ отвечает за:
 *
 * - сохранение в Supabase;
 * - создание Experience Skill;
 * - изменение существующего Experience;
 * - накопление Learning Candidates;
 * - Quality Gate;
 * - Approval;
 * - Autonomy Policy;
 * - версии Skill;
 * - запись Experience Memory.
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

    return (

        value &&

        typeof value === "object" &&

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
 * На текущем этапе Experience Analyzer
 * обучается только на успешных Execution.
 *
 * В будущем сюда НЕ нужно добавлять
 * логику Failure Learning.
 *
 * Когда Analyzer научится анализировать
 * неудачные выполнения, ограничение
 * по completed должно быть перенесено
 * непосредственно в Analyzer.
 *
 * =========================================================
 */


function shouldAnalyze(
    trace
) {


    if(
        !isObject(
            trace
        )
    ){

        return false;

    }



    /*
     * Текущий Experience Analyzer
     * требует хотя бы одно
     * успешное выполнение.
     */


    const completed =

        normalizeNumber(

            trace?.stats?.completed

        );



    if(
        completed <= 0
    ){

        return false;

    }



    return true;

}





/*
 * =========================================================
 * NORMALIZE PAYLOAD
 * =========================================================
 *
 * Learning Router является основным
 * источником Payload.
 *
 * Trigger обязан передать Payload дальше
 * без потери:
 *
 * - skillCandidate;
 * - skills;
 * - source;
 * - будущих Learning полей.
 *
 *
 * Fallback через Analysis нужен только
 * для устойчивости контракта.
 *
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
     * Candidate fallback
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
     * Existing Skills fallback
     *
     * Используется для
     * SKILL_IMPROVEMENT.
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
     * Source
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
 *
 * NEW_SKILL:
 *
 * payload.skillCandidate.skillId
 *
 *
 * SKILL_IMPROVEMENT:
 *
 * payload.skills[0].id
 * payload.skills[0].skillId
 *
 *
 * Fallback:
 *
 * decision.skillId
 *
 * =========================================================
 */


function resolveSkillId({

    decision,

    payload

}) {


    /*
     * NEW SKILL
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
     * EXISTING SKILL
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



    /*
     * ROUTER FALLBACK
     */


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
            Number.isFinite(
                number
            )
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
 * BUILD LEARNING EVENT
 * =========================================================
 *
 * Learning Event является транспортным
 * контрактом между:
 *
 * Analyzer / Router
 *        ↓
 * Learning Queue
 *        ↓
 * Worker
 *        ↓
 * Learning Proposal
 *
 *
 * Поэтому Event обязан содержать
 * полный Payload Router.
 *
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
         * =================================================
         * IDENTITY
         * =================================================
         */


        id:

            trace?.id ||

            null,



        traceId:

            trace?.id ||

            null,



        /*
         * =================================================
         * LEARNING DECISION
         * =================================================
         */


        action:

            decision?.action ||

            analysis?.action ||

            "IGNORE",



        skillId,



        confidence,



        reusable,



        reason,



        /*
         * =================================================
         * LEARNING PAYLOAD
         * =================================================
         *
         * КРИТИЧНО:
         *
         * Queue и Proposal читают Candidate
         * именно отсюда.
         *
         * NEW_SKILL:
         *
         * payload.skillCandidate
         *
         *
         * SKILL_IMPROVEMENT:
         *
         * payload.skillCandidate
         * payload.skills
         *
         * =================================================
         */


        payload,



        /*
         * =================================================
         * DIAGNOSTIC CONTEXT
         * =================================================
         *
         * Analysis и Decision сохраняются
         * для диагностики, Learning History
         * и будущего переанализа.
         *
         * Но следующие слои не должны
         * извлекать Candidate из
         * decision.payload.
         *
         * Канонический путь:
         *
         * event.payload
         *
         * =================================================
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
 * EMPTY RESULT
 * =========================================================
 */


function buildEmptyResult(
    reason
) {


    return {


        triggered:
            false,



        reason:

            normalizeText(
                reason
            )

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
 * ERROR RESULT
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

            trace?.id ||

            null


    };

}





/*
 * =========================================================
 * LOG EVENT
 * =========================================================
 */


function logLearningEvent(
    learningEvent
) {


    console.log(

        "Jessica Learning Event:",

        JSON.stringify({

            action:

                learningEvent?.action ||
                "IGNORE",



            skillId:

                learningEvent?.skillId ||
                null,



            confidence:

                learningEvent?.confidence ||
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

                    ?

                    learningEvent
                        .payload
                        .skills
                        .length

                    :

                    0

        })

    );

}





/*
 * =========================================================
 * RUN LEARNING TRIGGER
 * =========================================================
 */


export function runLearningTrigger(
    trace
) {


    /*
     * =====================================================
     * 1. CHECK TRACE
     * =====================================================
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
         * =================================================
         * 2. ANALYZE EXECUTION EXPERIENCE
         * =================================================
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
         * =================================================
         * 3. ROUTE LEARNING ACTION
         * =================================================
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
         * =================================================
         * 4. BUILD LEARNING EVENT
         * =================================================
         */


        const learningEvent =

            buildLearningEvent({

                trace,

                analysis,

                decision

            });



        /*
         * =================================================
         * 5. LOG
         * =================================================
         */


        logLearningEvent(
            learningEvent
        );



        /*
         * =================================================
         * 6. RESULT
         * =================================================
         */


        return {


            triggered:
                true,



            analysis,



            decision,



            learningEvent,



            traceId:

                trace?.id ||

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
