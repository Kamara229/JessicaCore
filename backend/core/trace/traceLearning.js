/*
 * =========================================================
 * JESSICA TRACE LEARNING v2
 * =========================================================
 *
 * Подготовка Execution Trace
 * для Learning слоя.
 *
 *
 * Ответственность:
 *
 * - сформировать Learning Payload;
 * - сохранить canonical Experience Usage;
 * - сохранить compatibility contract.
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
 * SAFE OBJECT
 * =========================================================
 */


function safeObject(
    value
) {

    return (

        value
        &&
        typeof value === "object"
        &&
        !Array.isArray(value)

    )

        ? value

        : {};

}


/*
 * =========================================================
 * BUILD LEARNING PAYLOAD
 * =========================================================
 */


export function buildLearningPayload(
    trace
) {

    if(
        !trace
    ){

        return null;

    }


    const experienceUsage = {

        ...safeObject(
            trace.experienceUsage
        )

    };


    return {

        /*
         * =================================================
         * IDENTITY
         * =================================================
         */


        executionId:
            trace.id,


        /*
         * Канонический traceId нужен
         * Candidate Memory для дедупликации
         * evidence.
         */


        traceId:
            trace.id,


        /*
         * =================================================
         * EXECUTION
         * =================================================
         */


        task:
            trace.task,


        status:
            trace.status,


        statistics:
            trace.statistics,


        /*
         * =================================================
         * EXPERIENCE
         * =================================================
         */


        experienceUsage,


        /*
         * Compatibility alias.
         *
         * Старые Learning-модули могут
         * пока читать payload.experience.
         *
         * После полной миграции alias
         * можно будет удалить.
         */


        experience:
            experienceUsage,


        /*
         * =================================================
         * EXECUTION EVIDENCE
         * =================================================
         */


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
