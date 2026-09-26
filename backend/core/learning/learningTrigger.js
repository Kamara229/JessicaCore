/*
 * =========================================================
 * JESSICA LEARNING TRIGGER
 * =========================================================
 *
 * Автоматический запуск анализа обучения
 * после выполнения задачи.
 *
 *
 * Flow:
 *
 * ExecutionTrace
 *       ↓
 * analyzeExecutionTrace()
 *       ↓
 * Learning Decision
 *
 *
 * НЕ:
 *
 * - сохраняет Experience;
 * - изменяет Supabase;
 * - создаёт Skill напрямую.
 *
 * Только запускает анализ.
 *
 * =========================================================
 */


import {
    analyzeExecutionTrace
} from "./experienceAnalyzer.js";



/*
 * =========================================================
 * SHOULD ANALYZE
 * =========================================================
 */


function shouldAnalyze(
    trace
) {


    if (
        !trace
    ) {

        return false;

    }



    /*
     * Есть выполненные шаги
     */

    if (
        !trace.stats ||
        trace.stats.completed === 0
    ) {

        return false;

    }



    return true;

}





/*
 * =========================================================
 * RUN LEARNING CHECK
 * =========================================================
 */


export function runLearningTrigger(
    trace
) {


    /*
     * Защита
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
                "Недостаточно данных для обучения",


            analysis:
                null


        };

    }





    try {


        const analysis =
            analyzeExecutionTrace(
                trace
            );



        return {


            triggered:
                true,


            analysis


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
                "Ошибка анализа обучения",


            error:
                error.message,


            analysis:
                null


        };

    }


}
