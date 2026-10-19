import {selectOperativeSourceText} from "./operativeSourceScope.js";

/*
 * Deterministic positive-evidence probe for explicit redistribution duties.
 * It does not infer that any duty is absent and it never certifies an answer.
 * A positive match means that "cannot verify" warrants another look.
 */
function operativeWords(source) {
  return selectOperativeSourceText(source)
    .replace(/\[([^\]]+)\]\([^)]*\)/g,"$1")
    .replace(/[*_`<>]/g," ")
    .replace(/\s+/g," ")
    .slice(0,60000);
}

export function containsExplicitNoticeDuty(source) {
  const text=operativeWords(source);
  return /\b(?:must|shall)\s+(?:retain|reproduce|include|preserve)\b[^.]{0,290}\b(?:copyright|attribution|notices?)\b/iu.test(text);
}
