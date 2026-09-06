/*
 * =========================================================
 * JESSICA RUN FAILURE POLICY
 * =========================================================
 *
 * Преобразует ошибки TaskRunner
 * в решение для Execution Cycle.
 *
 * TaskRunner сообщает ЧТО произошло.
 *
 * Этот модуль определяет:
 *
 * - можно ли перестроить план;
 * - какую причину передать Replanner;
 * - нужно ли завершить выполнение.
 *
 * Здесь нет вызова Replanner.
 */


/*
 * =========================================================
 * ANALYZE RUN FAILURE
 * =========================================================
 */


export function analyzeRunFailure(
    taskRunResult
) {

    /*
     * TaskRunner успешно завершился.
     */
    if (
        taskRunResult?.success === true
    ) {

        return {
            failed:
                false,

            shouldRetry:
                false
        };

    }


    /*
     * Требуется информация от пользователя.
     *
     * Replanner здесь не поможет.
     */
    if (
        taskRunResult?.needsClarification === true
    ) {

        return {
            failed:
                true,

            shouldRetry:
                false,

            needsClarification:
                true,

            stage:
                taskRunResult?.stage ||
                "tools",

            failureType:
                taskRunResult?.failureType ||
                "needs-clarification",

            reason:
                taskRunResult?.text ||
                "Для выполнения задачи требуется уточнение"
        };

    }


    /*
     * =====================================================
     * SEMANTIC ROUTE FAILURE
     * =====================================================
     *
     * TaskRunner сам определил,
     * что другой план может решить проблему.
     *
     * Например:
     *
     * - поиск не дал источников;
     * - Source Selector отклонил всю выдачу;
     * - текущий поисковый маршрут непригоден.
     */


    if (
        taskRunResult?.shouldRetry === true
    ) {

        return {
            failed:
                true,

            shouldRetry:
                true,

            needsClarification:
                false,

            stage:
                taskRunResult?.stage ||
                "runner",

            failureType:
                taskRunResult?.failureType ||
                "retryable-run-failure",

            reason:
                taskRunResult?.text ||
                "Текущий план не позволил получить подходящий результат"
        };

    }


    /*
     * =====================================================
     * NON-RETRYABLE FAILURE
     * =====================================================
     */


    return {
        failed:
            true,

        shouldRetry:
            false,

        needsClarification:
            false,

        stage:
            taskRunResult?.stage ||
            "tools",

        failureType:
            taskRunResult?.failureType ||
            "run-failure",

        reason:
            taskRunResult?.text ||
            "Не удалось выполнить план"
    };

}


/*
 * =========================================================
 * BUILD REPLANNER FEEDBACK
 * =========================================================
 *
 * Replanner уже умеет получать объект,
 * похожий на результат Validator:
 *
 * {
 *   valid,
 *   shouldRetry,
 *   reason
 * }
 *
 * Поэтому ошибки маршрута приводим
 * к тому же универсальному формату.
 */


export function buildRunFailureFeedback(
    analysis
) {

    return {
        success:
            true,

        valid:
            false,

        shouldRetry:
            analysis?.shouldRetry === true,

        needsClarification:
            analysis?.needsClarification === true,

        stage:
            analysis?.stage ||
            "runner",

        failureType:
            analysis?.failureType ||
            "run-failure",

        reason:
            analysis?.reason ||
            "Текущий план выполнения оказался непригодным"
    };

}
