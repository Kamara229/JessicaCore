/*
 * =========================================================
 * JESSICA TRACE LEARNING v3
 * =========================================================
 *
 * Подготовка Execution Trace
 * для Learning слоя.
 *
 *
 * Канонически сохраняет:
 *
 * - Experience Usage;
 * - реальные выполненные Tools;
 * - Execution Evidence;
 * - Statistics.
 *
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


function safeArray(
    value
) {

    return Array.isArray(value)

        ? [
            ...value
        ]

        : [];

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


    const executedTools =

        safeArray(
            trace.executedTools
        );


    return {

        /*
         * IDENTITY
         */


        executionId:
            trace.id,


        traceId:
            trace.id,


        /*
         * EXECUTION
         */


        task:
            trace.task,


        status:
            trace.status,


        statistics:
            trace.statistics,


        /*
         * EXPERIENCE
         */


        experienceUsage,


        /*
         * Compatibility alias.
         */


        experience:
            experienceUsage,


        /*
         * ACTUAL TOOL EVIDENCE
         *
         * Это реальные выполненные Tools,
         * а не инструменты из Planner Plan.
         */


        executedTools,


        /*
         * EXECUTION EVIDENCE
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
