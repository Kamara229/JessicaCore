/*
 * =========================================================
 * JESSICA LEARNING TRIGGER
 * =========================================================
 *
 * Точка входа Learning Pipeline.
 *
 *
 * Flow:
 *
 * ExecutionTrace
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
 * новый навык
 *
 *
 * SKILL_IMPROVEMENT
 *      ↓
 * новая версия существующего навыка
 *
 *
 * IGNORE
 *      ↓
 * опыт не сохраняем
 *
 *
 * НЕ отвечает за:
 *
 * - сохранение в Supabase;
 * - создание Skill;
 * - изменение Experience;
 * - Approval;
 * - версии.
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
 * SHOULD ANALYZE
 * =========================================================
 */


function shouldAnalyze(
    trace
) {


    if (
        !trace ||
        typeof trace !== "object"
    ) {

        return false;

    }



    /*
     * Нет успешного выполнения
     */


    if (
        !trace.stats ||
        trace.stats.completed <= 0
    ) {

        return false;

    }



    return true;

}





/*
 * =========================================================
 * BUILD LEARNING EVENT
 * =========================================================
 *
 * Унифицированный объект обучения.
 *
 * Передаётся дальше в:
 *
 * Queue
 * Approval
 * Storage
 *
 * =========================================================
 */


function buildLearningEvent({

    trace,

    analysis,

    decision

}) {


    return {


        id:
            trace?.id || null,


        action:
            decision?.action ||
            "IGNORE",



        skillId:

            decision?.skillId ||
            null,



        confidence:

            Number(
                decision?.confidence || 0
            ),



        traceId:

            trace?.id ||
            null,



        analysis,



        decision,



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


        reason,


        analysis:
            null,


        decision:
            null,


        learningEvent:
            null


    };


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
     * CHECK
     * =====================================================
     */


    if (
        !shouldAnalyze(
            trace
        )
    ) {


        return buildEmptyResult(

            "Недостаточно данных для обучения"

        );

    }





    try {



        /*
         * =================================================
         * 1. ANALYZE
         * =================================================
         */


        const analysis =
            analyzeExecutionTrace(
                trace
            );





        /*
         * =================================================
         * 2. ROUTE
         * =================================================
         */


        const decision =
            routeLearningEvent({

                trace,

                analysis

            });





        /*
         * =================================================
         * 3. EVENT
         * =================================================
         */


        const learningEvent =
            buildLearningEvent({

                trace,

                analysis,

                decision

            });





        console.log(

            "Jessica Learning Event:",

            JSON.stringify(

                {

                    action:
                        learningEvent.action,


                    skillId:
                        learningEvent.skillId,


                    confidence:
                        learningEvent.confidence

                }

            )

        );





        /*
         * =================================================
         * RESULT
         * =================================================
         */


        return {


            triggered:
                true,


            analysis,


            decision,


            learningEvent,


            traceId:
                trace.id || null


        };



    } catch(error) {



        console.error(

            "Jessica Learning Trigger error:",

            error

        );



        return {


            triggered:
                false,


            reason:
                "Ошибка Learning Pipeline",


            error:
                error?.message ||
                "unknown error",


            analysis:
                null,


            decision:
                null,


            learningEvent:
                null


        };


    }


}
