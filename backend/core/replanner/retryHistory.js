/*
 * =========================================================
 * JESSICA REPLANNER
 * RETRY HISTORY
 * =========================================================
 *
 * Собирает компактную историю предыдущих попыток
 * выполнения задачи.
 *
 *
 * Используется Replanner для понимания:
 *
 * - какой intent уже использовался;
 * - какие инструменты уже запускались;
 * - какие поисковые запросы уже пробовали;
 * - какой маршрут уже оказался неэффективным.
 *
 *
 * Этот модуль НЕ:
 *
 * - вызывает Planner;
 * - вызывает AI;
 * - создаёт инструкции;
 * - изменяет PlanningContext;
 * - выполняет инструменты.
 *
 * =========================================================
 */


/*
 * =========================================================
 * CONFIG
 * =========================================================
 */


const MAX_TRIED_SEARCH_QUERIES =
    8;


/*
 * =========================================================
 * SAFE STRING
 * =========================================================
 */


function safeString(
    value
) {

    return typeof value === "string"
        ? value.trim()
        : "";

}


/*
 * =========================================================
 * UNIQUE STRINGS
 * =========================================================
 */


function uniqueStrings(
    values
) {

    if (
        !Array.isArray(
            values
        )
    ) {

        return [];

    }


    return [

        ...new Set(

            values

                .map(
                    item =>
                        safeString(
                            item
                        )
                )

                .filter(Boolean)

        )

    ];

}


/*
 * =========================================================
 * EXTRACT SEARCH QUERIES
 * =========================================================
 *
 * Извлекает web_search query из плана.
 *
 *
 * Например:
 *
 * [
 *   {
 *     tool: "web_search",
 *     arguments: {
 *       query: "Godot Engine official website"
 *     }
 *   }
 * ]
 *
 * →
 *
 * [
 *   "Godot Engine official website"
 * ]
 *
 * =========================================================
 */


export function extractSearchQueries(
    plan
) {

    if (
        !Array.isArray(
            plan?.steps
        )
    ) {

        return [];

    }


    const queries =
        plan.steps

            .filter(
                step =>
                    step?.tool === "web_search"
            )

            .map(
                step =>
                    safeString(
                        step
                            ?.arguments
                            ?.query
                    )
            );


    return uniqueStrings(
        queries
    );

}


/*
 * =========================================================
 * EXTRACT USED TOOLS
 * =========================================================
 *
 * Берём только реально выполненные инструменты
 * из TaskRunner results.
 *
 * =========================================================
 */


export function extractUsedTools(
    runResult
) {

    if (
        !Array.isArray(
            runResult?.results
        )
    ) {

        return [];

    }


    return uniqueStrings(

        runResult.results.map(
            item =>
                item?.tool
        )

    );

}


/*
 * =========================================================
 * EXTRACT FAILED TOOLS
 * =========================================================
 *
 * Если конкретный Tool успел выполниться
 * и вернул success=false,
 * сохраняем его отдельно.
 *
 *
 * Важно:
 *
 * при Source Selector reject здесь может быть [],
 * потому что web_fetch ещё не запускался.
 *
 * Это нормальная ситуация.
 *
 * =========================================================
 */


export function extractFailedTools(
    runResult
) {

    if (
        !Array.isArray(
            runResult?.results
        )
    ) {

        return [];

    }


    return uniqueStrings(

        runResult.results

            .filter(
                item =>
                    item?.success === false
            )

            .map(
                item =>
                    item?.tool
            )

    );

}


/*
 * =========================================================
 * BUILD PREVIOUS ATTEMPT
 * =========================================================
 *
 * Компактное описание предыдущей попытки.
 *
 * Полный plan и полный TaskRunner result
 * сюда намеренно не копируем.
 *
 * =========================================================
 */


export function buildPreviousAttempt(
    previousPlan,
    previousRunResult
) {

    return {

        intent:
            safeString(
                previousPlan?.intent
            ),


        stepsCount:
            Array.isArray(
                previousPlan?.steps
            )
                ? previousPlan.steps.length
                : 0,


        searchQueries:
            extractSearchQueries(
                previousPlan
            ),


        usedTools:
            extractUsedTools(
                previousRunResult
            ),


        failedTools:
            extractFailedTools(
                previousRunResult
            )

    };

}


/*
 * =========================================================
 * BUILD TRIED SEARCH QUERIES
 * =========================================================
 *
 * Накапливает поисковые запросы между Replan.
 *
 *
 * Например:
 *
 * Attempt 1:
 *
 * "Quasar Banana Engine ZX-9100 official website"
 *
 *
 * Attempt 2:
 *
 * "Quasar Banana Engine ZX-9100 developer"
 *
 *
 * После этого история:
 *
 * [
 *   "Quasar Banana Engine ZX-9100 official website",
 *   "Quasar Banana Engine ZX-9100 developer"
 * ]
 *
 *
 * Это позволяет Planner не ходить по кругу.
 *
 * =========================================================
 */


export function buildTriedSearchQueries(
    planningContext,
    previousPlan
) {

    const existing =
        Array.isArray(
            planningContext
                ?.metadata
                ?.triedSearchQueries
        )
            ? planningContext
                .metadata
                .triedSearchQueries
            : [];


    const current =
        extractSearchQueries(
            previousPlan
        );


    return uniqueStrings([

        ...existing,

        ...current

    ])
        .slice(
            -MAX_TRIED_SEARCH_QUERIES
        );

}


/*
 * =========================================================
 * BUILD RETRY HISTORY
 * =========================================================
 *
 * Единая точка входа для Replanner.
 *
 * =========================================================
 */


export function buildRetryHistory({

    planningContext = {},

    previousPlan = null,

    previousRunResult = null

} = {}) {


    return {

        previousAttempt:
            buildPreviousAttempt(

                previousPlan,

                previousRunResult

            ),


        triedSearchQueries:
            buildTriedSearchQueries(

                planningContext,

                previousPlan

            )

    };

}
