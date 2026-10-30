import {extractInvalidCalendarRange} from "../native/invalidCalendarRange.js";

/** Do not learn success from a nonexistent user-supplied date. */
export function skipInvalidCalendarInputLearning(trace){
 const tools=Array.isArray(trace?.executedTools)?trace.executedTools:[];
 return tools.length===1 && tools[0]==="invalid_calendar_date" &&
   Boolean(extractInvalidCalendarRange(trace?.task));
}
