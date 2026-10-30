/** Jessica response facade for single and complex answers. */
import { buildSingleTaskResponse } from "./singleTaskResponse.js";
import { buildComplexTaskResponse } from "./complexTaskResponse.js";
import { applyStrictNumericOnlyFormat } from "./strictNumericOnlyResponse.js";

export async function buildJessicaResponse({
    task, decomposition, executionResult, executionTrace
}) {
    const response = Array.isArray(executionResult?.results)
        ? await buildComplexTaskResponse(task, decomposition, executionResult, executionTrace)
        : buildSingleTaskResponse(executionResult, decomposition, executionTrace);
    return applyStrictNumericOnlyFormat(task, response);
}
