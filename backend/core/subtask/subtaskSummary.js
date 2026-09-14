/*
 * =========================================================
 * JESSICA SUBTASK SUMMARY
 * =========================================================
 *
 * Подсчитывает итоговую статистику
 * по результатам выполнения подзадач.
 *
 *
 * Этот модуль НЕ:
 *
 * - выполняет подзадачи;
 * - вызывает Planner;
 * - работает с Experience;
 * - вызывает инструменты;
 * - формирует пользовательский ответ.
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
        Array.isArray(
            results
        )
            ? results
            : [];


    /*
     * =====================================================
     * COMPLETED
     * =====================================================
     */


    const completed =
        safeResults.filter(
            item =>
                item?.status ===
                "COMPLETED"
        ).length;


    /*
     * =====================================================
     * NEEDS CLARIFICATION
     * =====================================================
     */


    const needsClarification =
        safeResults.filter(
            item =>
                item?.status ===
                "NEEDS_CLARIFICATION"
        ).length;


    /*
     * =====================================================
     * FAILED
     * =====================================================
     */


    const failed =
        safeResults.filter(
            item =>
                item?.status ===
                "FAILED"
        ).length;


    /*
     * =====================================================
     * RESULT
     * =====================================================
     */


    return {

        success:
            completed > 0,

        total:
            safeResults.length,

        completed,

        needsClarification,

        failed,

        results:
            safeResults

    };


}
