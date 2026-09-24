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
 * - сохранение результатов.
 *
 *
 * НЕ:
 *
 * - выполняет задачи;
 * - вызывает AI;
 * - работает с Experience;
 * - формирует ответ.
 *
 * =========================================================
 */



export function buildSubtaskSummary(
    results = []
) {


    const safeResults =
        Array.isArray(results)
            ? results
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
            item?.status
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
         * Есть ли хотя бы один
         * полезный результат
         */


        success:
            summary.completed > 0,



        /*
         * Полная статистика
         */


        ...summary,



        /*
         * Сырые результаты
         */


        results:
            safeResults



    };


}
