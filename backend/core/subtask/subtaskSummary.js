/*
 * =========================================================
 * JESSICA SUBTASK SUMMARY
 * =========================================================
 *
 * Аггрегация результатов выполнения подзадач.
 *
 *
 * Отвечает только за:
 *
 * - подсчёт статистики;
 * - возврат списка результатов.
 *
 *
 * НЕ:
 *
 * - выполняет задачи;
 * - вызывает AI;
 * - работает с Experience;
 * - работает с Tools;
 * - формирует ответ.
 *
 * =========================================================
 */





/*
 * =========================================================
 * BUILD SUMMARY
 * =========================================================
 */


export function buildSubtaskSummary(
    results = []
) {


    const safeResults =
        Array.isArray(results)
            ? results.filter(Boolean)
            : [];





    const summary = {


        total:
            safeResults.length,



        completed:
            0,



        needsClarification:
            0,



        failed:
            0,



        unknown:
            0


    };








    for (
        const item
        of safeResults
    ) {


        switch(
            item.status
        ) {



            case "COMPLETED":


                summary.completed++;

                break;





            case "NEEDS_CLARIFICATION":


                summary.needsClarification++;

                break;





            case "FAILED":


                summary.failed++;

                break;





            default:


                summary.unknown++;

                break;


        }


    }









    return {


        /*
         * Есть ли полезный результат
         */


        success:
            summary.completed > 0,



        /*
         * Есть ли вообще результаты
         */


        hasResults:
            summary.total > 0,



        /*
         * Статистика
         */


        ...summary,



        /*
         * Результаты подзадач
         */


        results:
            safeResults


    };


}
