/*
 * =========================================================
 * JESSICA TRACE LEARNING v1
 * =========================================================
 *
 * Подготовка Execution Trace
 * для Learning слоя.
 *
 *
 * Ответственность:
 *
 * - сформировать Learning Payload.
 *
 *
 * НЕ:
 *
 * - анализирует Learning;
 * - создаёт Skill;
 * - сохраняет Experience;
 * - изменяет Trace.
 *
 * =========================================================
 */



/*
 * =========================================================
 * BUILD LEARNING PAYLOAD
 * =========================================================
 */


export function buildLearningPayload(

    trace

){


    if(!trace)
        return null;



    return {


        executionId:

            trace.id,


        task:

            trace.task,


        status:

            trace.status,


        statistics:

            trace.statistics,


        experience:

            trace.experienceUsage,


        failures:

            trace.failures,


        replans:

            trace.replans,


        terminal:

            trace.terminal,


        validation:

            trace.validation,


        steps:

            trace.steps,


        events:

            trace.events,


        result:

            trace.result

    };

}
