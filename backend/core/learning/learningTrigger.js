/*
 * =========================================================
 * JESSICA LEARNING TRIGGER
 * =========================================================
 *
 * Автоматический запуск системы обучения
 * после выполнения задачи.
 *
 *
 * Flow:
 *
 * ExecutionTrace
 *       ↓
 * Experience Analyzer
 *       ↓
 * Learning Router
 *       ↓
 * Learning Decision
 *
 *
 * Возможные решения:
 *
 * NEW_SKILL
 *      ↓
 * создание нового навыка
 *
 *
 * SKILL_IMPROVEMENT
 *      ↓
 * улучшение существующего навыка
 *
 *
 * IGNORE
 *      ↓
 * опыт не сохраняем
 *
 *
 * Этот модуль НЕ:
 *
 * - сохраняет данные;
 * - изменяет Supabase;
 * - создаёт Skill;
 * - обновляет Experience.
 *
 * Только анализирует и выбирает направление обучения.
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
     * Нет выполненных действий
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


        return {


            triggered:
                false,


            reason:
                "Недостаточно данных для анализа обучения",


            analysis:
                null,


            decision:
                null


        };

    }





    try {



        /*
         * =================================================
         * 1. ANALYZE EXPERIENCE
         * =================================================
         */


        const analysis =
            analyzeExecutionTrace(
                trace
            );





        /*
         * =================================================
         * 2. ROUTE LEARNING
         * =================================================
         *
         * Router решает:
         *
         * создать новый Skill
         * или улучшить старый
         *
         * =================================================
         */


        const decision =
            routeLearningEvent({

                trace,

                analysis

            });





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
                "Ошибка запуска Learning Pipeline",


            error:
                error?.message ||
                "unknown error",


            analysis:
                null,


            decision:
                null


        };


    }


}
