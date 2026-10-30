import {dateCalculationExplanationMissing} from "../validator/dateCalculationExplanation.js";
/** Never reinforce native date Skill for missing a requested explanation. */
export function missingNativeDateExplanation(trace) {
  const task=String(trace?.task||"").toLowerCase();

  const tools=[
    ...(Array.isArray(trace?.executedTools)?trace.executedTools:[]),
    ...(Array.isArray(trace?.result?.executionMeta?.usedTools)?
      trace.result.executionMeta.usedTools:[])
  ];
  if (!tools.includes("date_diff")) return false;
  const answer=String(trace?.result?.answer?.text||trace?.result?.text||
    trace?.result?.outputText||"");
  return dateCalculationExplanationMissing(task,answer);
}
