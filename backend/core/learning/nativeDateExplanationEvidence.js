/** Never reinforce native date Skill for missing a requested explanation. */
export function missingNativeDateExplanation(trace) {
  const task=String(trace?.task||"").toLowerCase();
  if (!/(объясн|explain)/iu.test(task)||!/(дн|day)/iu.test(task)) return false;
  const tools=[
    ...(Array.isArray(trace?.executedTools)?trace.executedTools:[]),
    ...(Array.isArray(trace?.result?.executionMeta?.usedTools)?
      trace.result.executionMeta.usedTools:[])
  ];
  if (!tools.includes("date_diff")) return false;
  const answer=String(trace?.result?.text||trace?.result?.answer||
    trace?.result?.outputText||"");
  return !/(расч[её]т|учитыва(?:ем|ется)|считаем|count(?:ing)?|excluding)/iu.test(answer);
}
