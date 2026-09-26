/*
 * =========================================================
 * JESSICA LEARNING COORDINATOR
 * =========================================================
 *
 * Центральный координатор обучения.
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
 * НЕ:
 *
 * - сохраняет Experience;
 * - пишет в Supabase;
 * - подтверждает обучение.
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
 * PROCESS LEARNING
 * =========================================================
 */


export function processLearning(
    trace
) {


    /*
     * Запускаем анализ
     */


    const triggerResult =
        runLearningTrigger(
            trace
        );



    if (
        !triggerResult.triggered
    ) {


        return {


            success:
                true,


            queued:
                false,


            trigger:
                triggerResult


        };

    }





    const event =
        triggerResult.learningEvent;



    if (
        !event
    ) {


        return {


            success:
                false,


            queued:
                false,


            reason:
                "Learning Event не создан"


        };

    }





    /*
     * Создаём элемент очереди
     */


    const queueItem =
        createLearningQueueItem(
            event
        );





    if (
        !queueItem
    ) {


        return {


            success:
                false,


            queued:
                false,


            reason:
                "Не удалось создать Queue Item"


        };

    }





    return {


        success:
            true,


        queued:
            true,


        queueItem,


        trigger:
            triggerResult


    };


}
