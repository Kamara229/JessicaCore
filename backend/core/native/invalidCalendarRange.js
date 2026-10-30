import {parseNativeDate} from "./dateUtils.js";

/** Detect two explicitly supplied date operands, one or more invalid.
 * Neither invent corrected dates nor treat compound questions as one calculation.
 */
export function extractInvalidCalendarRange(task){
  const text=String(task||"").trim();
  if(text.length>500 ||
     !/^Сколько\s+дней(?:\s+прошло)?\s+между\s/iu.test(text) ||
     /(?:затем|далее|после этого|и сколько|и вычисли|плюс|добавь|then|compare)/iu.test(text))
    return null;
  const dates=[...text.matchAll(
    /(?<![\p{L}\p{N}])(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[.\/-]\d{1,2}[.\/-]\d{4}|\d{1,2}\s+[\p{L}]+\s+\d{4}(?:\s*г(?:ода|\.)?)?)(?![\p{L}\p{N}])/giu
  )];
  if(dates.length!==2)return null;
  const before=text.slice(0,dates[0].index);
  const between=text.slice(dates[0].index+dates[0][0].length,dates[1].index);
  const after=text.slice(dates[1].index+dates[1][0].length);
  if(!/^Сколько\s+дней(?:\s+прошло)?\s+между\s*$/iu.test(before) ||
     !/^\s*и\s*$/iu.test(between) ||
     !/^\s*(?:\?\s*(?:Покажи\s+расч[её]т\s*\.?)?)?\s*$/iu.test(after))return null;
  const operands=dates.map(match=>match[0]);
  const invalidDates=operands.filter(value=>!parseNativeDate(value));
  if(invalidDates.length===0)return null;
  return {startDate:operands[0],endDate:operands[1],invalidDates};
}
