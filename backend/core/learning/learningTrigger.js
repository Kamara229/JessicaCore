/*
 * =========================================================
 * JESSICA LEARNING TRIGGER FACADE v7
 * =========================================================
 *
 * Execution Trace
 *      ↓
 * eligibility
 *      ↓
 * multi-skill analysis
 *      ↓
 * Learning Router
 *      ↓
 * one or more Learning Events
 *
 * Compatibility:
 * learningEvent / analysis / decision still expose
 * the first event for old callers.
 * =========================================================
 */


import {
    routeLearningEvent
} from "./learningRouter.js";


import {
    analyzeExecutionTraceBatch
} from "./multiSkill/multiSkillAnalyzer.js";


import {
    shouldAnalyzeLearningTrace
} from "./trigger/triggerEligibility.js";


import {
    buildLearningEvent
} from "./trigger/learningEventBuilder.js";


import {
    logLearningEvents
} from "./trigger/learningEventLogger.js";


function buildEmptyResult(
    reason
) {

    return {

        triggered:
            false,

        reason:

            String(
                reason || "Learning не запущен"
            )
                .trim(),

        analyses:
            [],

        decisions:
            [],

        learningEvents:
            [],

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


function buildErrorResult(
    trace,
    error
) {

    return {

        ...buildEmptyResult(
            "Ошибка Learning Pipeline"
        ),

        error:

            String(
                error?.message || "unknown error"
            )
                .trim(),

        traceId:

            trace?.id

            ||

            trace?.traceId

            ||

            null

    };

}


export function runLearningTrigger(
    trace
) {

    if(
        !shouldAnalyzeLearningTrace(
            trace
        )
    ){

        return buildEmptyResult(
            "Недостаточно данных для обучения"
        );

    }


    try {

        const analyses =

            analyzeExecutionTraceBatch(
                trace
            )
                .filter(
                    item =>
                        item
                        &&
                        typeof item === "object"
                        &&
                        !Array.isArray(item)
                );


        if(
            analyses.length === 0
        ){

            return buildEmptyResult(
                "Experience Analyzer не вернул результат"
            );

        }


        const decisions = [];

        const learningEvents = [];


        for(
            const analysis
            of analyses
        ){

            const decision =

                routeLearningEvent({

                    trace,

                    analysis

                });


            if(
                !decision
                ||
                typeof decision !== "object"
            ){

                continue;

            }


            decisions.push(
                decision
            );


            learningEvents.push(

                buildLearningEvent({

                    trace,

                    analysis,

                    decision

                })

            );

        }


        if(
            learningEvents.length === 0
        ){

            return buildEmptyResult(
                "Learning Router не вернул решений"
            );

        }


        logLearningEvents(
            learningEvents
        );


        return {

            triggered:
                true,

            analyses,

            decisions,

            learningEvents,

            analysis:
                analyses[0] || null,

            decision:
                decisions[0] || null,

            learningEvent:
                learningEvents[0] || null,

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


        return buildErrorResult(
            trace,
            error
        );

    }

}
