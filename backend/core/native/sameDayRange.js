import {parseNativeDate} from "./dateUtils.js";
// Preserve two explicit occurrences even when they denote the same date.
// extractNativeDates intentionally deduplicates, so it cannot represent this case.
export function extractSameDayRange(task) {
  const text=String(task||"");
  if(text.length>500 || !/(?:между|\bс\b|\bот\b|between|from)/iu.test(text))return null;
  const matches=[...text.matchAll(/\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[.\/-]\d{1,2}[.\/-]\d{4})\b/gu)];
  if(matches.length!==2)return null;
  const first=parseNativeDate(matches[0][0]),second=parseNativeDate(matches[1][0]);
  if(!first || !second || first.iso!==second.iso)return null;
  const between=text.slice(matches[0].index+matches[0][0].length,matches[1].index);
  if(!/^\s*(?:и|по|до|and|to)\s*$/iu.test(between))return null;
  return {startDate:first.iso,endDate:second.iso};
}