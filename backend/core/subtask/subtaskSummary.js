/*
 * =========================================================
 * JESSICA SUBTASK SUMMARY v2
 * =========================================================
 *
 * Агрегация результатов выполнения подзадач.
 *
 * Отвечает только за:
 *
 * - подсчёт статистики;
 * - итоговый статус complex execution;
 * - возврат списка результатов.
 * =========================================================
 */


function normalizeStatus(
    value
) {

    return String(
        value || ""
    )
        .trim()
        .toUpperCase();

}


function resolveSummaryStatus(
    summary
) {

    if(
        summary.total > 0
        &&
        summary.completed === summary.total
    ){

        return "COMPLETED";

    }


    if(
        summary.completed > 0
    ){

        return "PARTIAL";

    }


    if(
        summary.needsClarification > 0
        &&
        summary.failed === 0
        &&
        summary.unknown === 0
    ){

        return "NEEDS_CLARIFICATION";

    }


    return "FAILED";

}


/*
 * =========================================================
 * BUILD SUMMARY
 * =========================================================
 */


export function buildSubtaskSummary(
    results = []
) {

    const safeResults =

        Array.isArray(
            results
        )

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

        blocked:
            0,

        unknown:
            0

    };


    for(
        const item
        of safeResults
    ){

        switch(
            normalizeStatus(
                item?.status
            )
        ){

            case "COMPLETED":

                summary.completed++;

                break;


            case "NEEDS_CLARIFICATION":

                summary.needsClarification++;

                break;


            case "BLOCKED":

                summary.blocked++;
                summary.failed++;

                break;


            case "FAILED":

            case "NO_VERIFIED_RESULT":

                summary.failed++;

                break;


            default:

                summary.unknown++;

                break;

        }

    }


    const status =

        resolveSummaryStatus(
            summary
        );


    return {

        success:

            status === "COMPLETED"

            ||

            status === "PARTIAL",

        status,

        hasResults:
            summary.total > 0,

        ...summary,

        results:
            safeResults

    };

}
