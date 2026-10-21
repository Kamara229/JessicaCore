import {analyzeProgrammingTask} from "./programmingTaskContext.js";

/** Never turn unexecuted, LLM-generated source into a validated programming Skill. */
const CODE_ARTIFACT=/(?:\b(?:write|implement|fix|debug|refactor|compile|build|patch|generate|create|code|function|script|class|unit tests?)\b|напиши|реализуй|исправь|отлад|рефактор|скомпил|сгенерир|создай (?:код|скрипт|класс|функц)|код|скрипт|функцию)/iu;
const VERIFIED_TOOLS=new Set(["run_code","compile_code","run_tests","code_test","program_test"]);

function runResults(value) {
    if (!value || typeof value!=="object") return [];
    const direct=Array.isArray(value.toolResults)?value.toolResults:[];
    const nested=Array.isArray(value.results)?value.results.flatMap(runResults):[];
    return [...direct,...nested];
}
export function needsProgrammingRuntimeEvidence(task,executionResult) {
    if (!analyzeProgrammingTask(task).relevant || !CODE_ARTIFACT.test(String(task||""))) return false;
    const tested=runResults(executionResult).some(item=>
        VERIFIED_TOOLS.has(String(item?.tool||item?.name||"")) &&
        item?.success===true && item?.verified===true &&
        item?.validation?.valid===true
    );
    return !tested;
}
