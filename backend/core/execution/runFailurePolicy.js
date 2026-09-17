/*
 * =========================================================
 * JESSICA RUN FAILURE POLICY
 * =========================================================
 *
 * Преобразует результат неудачного выполнения TaskRunner
 * в решение для Execution Cycle.
 *
 *
 * TaskRunner сообщает ЧТО произошло.
 *
 * Этот модуль определяет:
 *
 * - произошла ли ошибка;
 * - можно ли перестроить маршрут;
 * - требуется ли уточнение пользователя;
 * - какую причину передать Replanner.
 *
 *
 * Этот модуль НЕ:
 *
 * - выполняет инструменты;
 * - вызывает Replanner;
 * - создаёт новый план.
 *
 * =========================================================
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
     * =====================================================
     * SUCCESS
     * =====================================================
     */


    if (
        taskRunResult?.success === true
    ) {

        return {

            failed:
                false,

            shouldRetry:
                false,

            needsClarification:
                false

        };

    }



    /*
     * =====================================================
     * REASON
     * =====================================================
     */


    const reason =

        taskRunResult?.reason ||

        taskRunResult?.text ||

        "";



    /*
     * =====================================================
     * NEEDS CLARIFICATION
     * =====================================================
     *
     * Другой маршрут здесь не поможет.
     * Нужны дополнительные данные пользователя.
     *
     * =====================================================
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
                reason ||
                "Для выполнения задачи требуется уточнение"

        };

    }



    /*
     * =====================================================
     * RETRYABLE ROUTE FAILURE
     * =====================================================
     *
     * TaskRunner сообщил, что проблема может
     * быть решена другим маршрутом.
     *
     *
     * Например:
     *
     * - поиск ничего не дал;
     * - найденные источники не подходят;
     * - Source Selector отклонил результаты;
     * - fetch не позволил получить нужный источник;
     * - выбранный маршрут оказался непригодным.
     *
     * =====================================================
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
                reason ||
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
            reason ||
            "Не удалось выполнить план"

    };

}



/*
 * =========================================================
 * BUILD REPLANNER FEEDBACK
 * =========================================================
 */


export function buildRunFailureFeedback(
    analysis
) {

    return {

        stage:
            analysis?.stage ||
            "runner",


        failureType:
            analysis?.failureType ||
            "run-failure",


        reason:
            analysis?.reason ||
            "Текущий маршрут выполнения оказался непригодным",


        shouldRetry:
            analysis?.shouldRetry === true,


        needsClarification:
            analysis?.needsClarification === true

    };

}
