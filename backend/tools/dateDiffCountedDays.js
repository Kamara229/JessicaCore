/** Explain a short (start, end] calendar interval by the actual counted days. */
import {russianDayForm} from "./russianDayInflection.js";
const MS_PER_DAY=86400000;

export function countedCalendarDays(start,end,days) {
  if (!Number.isSafeInteger(days) || days<1) return "";
  if (days>10) return " Расчёт: ("+end+" − "+start+") = "+days+" "+russianDayForm(days).full+".";
  const first=Date.parse(start+"T00:00:00Z");
  if (!Number.isFinite(first)) return "";
  const dates=Array.from({length:days},(_,index)=>
    new Date(first+(index+1)*MS_PER_DAY).toISOString().slice(0,10));
  if (dates.at(-1)!==end) return "";
  return " Расчёт: учитываем "+dates.join(", ")+" — всего "+days+".";
}
