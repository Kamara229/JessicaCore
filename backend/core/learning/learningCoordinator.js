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
 * Learning Queue
 *
 *
 * Ответственность:
 *
 * - принять Execution Trace;
 * - запустить анализ обучения;
 * - получить Learning Decision;
 * - создать Queue Item.
 *
 *
 * НЕ отвечает за:
 *
 * - создание Skill;
 * - сохранение Experience;
 * - запись в Supabase;
 * - Approval;
 * - изменение памяти Jessica.
 *
 * =========================================================
 */



import {
    runLearningTrigger
} from "./learningTrigger.js";


import {
    createLearningQueueItem
} from "./learningQueue.js";





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
 * IGNORE RESULT
 * =========================================================
 */


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
 * FAILED RESULT
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





/*
 * =========================================================
 * PROCESS LEARNING
 * =========================================================
 */


export function processLearning(
    trace
) {


    /*
     * =====================================================
     * TRACE VALIDATION
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
     * LEARNING ANALYSIS
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

            "Jessica Learning Coordinator trigger error:",

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
     *
     * Не сохраняем бесполезный опыт.
     *
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

            "Jessica Learning Queue creation error:",

            error

        );



        return buildFailureResult(

            "Не удалось создать Learning Queue Item"

        );

    }





    if (
        !queueItem
    ) {


        return buildFailureResult(

            "Learning Queue Item пустой"

        );

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


        trigger:
            triggerResult


    };


}
