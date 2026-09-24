export function createExecutionTrace(task) {

    return {

        task,

        startedAt:
            new Date().toISOString(),

        completed:
            false,

        subtasks:
            [],

        usedTools:
            [],

        validationErrors:
            []

    };

}


export function updateTraceFromResult(
    trace,
    result
) {

    if (!trace || !result) {
        return trace;
    }


    trace.subtasks.push(
        result
    );


    trace.usedTools =
        result.usedTools || [];


    trace.completed =
        result.status === "COMPLETED";


    return trace;

}
