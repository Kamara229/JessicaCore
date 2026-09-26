/*
 * =========================================================
 * JESSICA LEARNING COORDINATOR
 * =========================================================
 *
 * Центральный координатор Learning Pipeline.
 *
 *
 * Flow:
 *
 * ExecutionTrace
 *        ↓
 * Learning Trigger
 *        ↓
 * Learning Event
 *        ↓
 * Learning Queue Builder
 *        ↓
 * Learning Queue Storage
 *        ↓
 * Supabase
 *
 *
 * Ответственность:
 *
 * - принять Execution Trace;
 * - запустить анализ обучения;
 * - получить Learning Decision;
 * - создать Queue Item;
 * - сохранить Queue Item.
 *
 *
 * НЕ отвечает за:
 *
 * - создание Skill;
 * - Experience Approval;
 * - изменение памяти Jessica;
 * - обучение напрямую.
 *
 * =========================================================
 */



import {
    runLearningTrigger
} from "./learningTrigger.js";


import {
    createLearningQueueItem
} from "./learningQueue.js";


import {
    saveLearningQueueItem
} from "./learningQueueStorage.js";





/*
 * =========================================================
 * VALIDATE TRACE
 * =========================================================
 */


function isValidTrace(
    trace
) {

    return (

        trace &&

        typeof trace === "object"

    );

}





/*
 * =========================================================
 * RESULT HELPERS
 * =========================================================
 */


function buildFailureResult(
    reason
) {

    return {

        success:
            false,

        type:
            "LEARNING_EVENT",

        queued:
            false,

        reason

    };

}





function buildIgnoredResult(
    triggerResult
) {

    return {

        success:
            true,

        type:
            "LEARNING_EVENT",

        queued:
            false,

        reason:
            "Learning decision IGNORE",

        trigger:
            triggerResult

    };

}





/*
 * =========================================================
 * PROCESS LEARNING
 * =========================================================
 */


export async function processLearning(
    trace
) {


    /*
     * =====================================================
     * VALIDATE
     * =====================================================
     */


    if (
        !isValidTrace(
            trace
        )
    ) {

        return buildFailureResult(
            "Invalid execution trace"
        );

    }





    /*
     * =====================================================
     * TRIGGER
     * =====================================================
     */


    let triggerResult;


    try {


        triggerResult =
            runLearningTrigger(
                trace
            );


    } catch(error) {


        console.error(
            "Jessica Learning Trigger error:",
            error
        );


        return buildFailureResult(
            "Ошибка запуска Learning Trigger"
        );

    }





    /*
     * =====================================================
     * NOTHING TO LEARN
     * =====================================================
     */


    if (
        !triggerResult?.triggered
    ) {


        return {

            success:
                true,

            type:
                "LEARNING_EVENT",

            queued:
                false,

            reason:
                triggerResult?.reason ||
                "Learning не запущен",

            trigger:
                triggerResult

        };

    }





    /*
     * =====================================================
     * GET EVENT
     * =====================================================
     */


    const learningEvent =
        triggerResult.learningEvent;



    if (
        !learningEvent
    ) {

        return buildFailureResult(
            "Learning Event не создан"
        );

    }





    /*
     * =====================================================
     * IGNORE
     * =====================================================
     */


    if (
        learningEvent.action ===
        "IGNORE"
    ) {


        return buildIgnoredResult(
            triggerResult
        );

    }





    /*
     * =====================================================
     * CREATE QUEUE ITEM
     * =====================================================
     */


    let queueItem;


    try {


        queueItem =
            createLearningQueueItem(
                learningEvent
            );


    } catch(error) {


        console.error(
            "Learning Queue Builder error:",
            error
        );


        return buildFailureResult(
            "Не удалось создать Queue Item"
        );

    }





    if (
        !queueItem
    ) {

        return buildFailureResult(
            "Queue Item пустой"
        );

    }





    /*
     * =====================================================
     * SAVE TO SUPABASE
     * =====================================================
     */


    let saveResult;


    try {


        saveResult =
            await saveLearningQueueItem(
                queueItem
            );


    } catch(error) {


        console.error(
            "Learning Queue Storage error:",
            error
        );


        return buildFailureResult(
            "Ошибка сохранения Learning Queue"
        );

    }





    if (
        !saveResult?.success
    ) {


        return {

            success:
                false,

            type:
                "LEARNING_EVENT",

            queued:
                false,

            queueItem,

            reason:
                saveResult?.error ||
                "Learning Queue не сохранён"

        };

    }





    /*
     * =====================================================
     * SUCCESS
     * =====================================================
     */


    return {


        success:
            true,


        type:
            "LEARNING_EVENT",


        queued:
            true,


        queueItem,


        storage:
            saveResult,


        trigger:
            triggerResult


    };


}
