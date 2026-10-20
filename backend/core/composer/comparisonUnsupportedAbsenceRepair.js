/*
 * Remove a separable, unproven absence statement from a comparison cell.
 * Silence of a primary source never establishes that a legal requirement
 * does not exist. Preserve the independently checkable positive statements.
 * Only a directly expressed negation of the requirement in the actual
 * operative source can retain an absence claim.
 */
import {selectOperativeSourceText} from "../validator/claim/operativeSourceScope.js";

const NEGATIVE=/(?:\bне\s+(?:требу\p{L}*|содерж\p{L}*|предусматрива\p{L}*|упомина\p{L}*|име\p{L}*)\b|\b(?:нет|отсутств\p{L}*)\b)/iu;
const EXPLICIT_SOURCE_NEGATION=/\b(?:not required|no requirement|no obligation|does not require|shall not require|must not require)\b/iu;

function separateClaimAndCitation(text){
  // Inline legal quotes/citations are not the claimed provision.
  const index=String(text).search(/[«\[]/u);
  return index<0?[text,""]:[text.slice(0,index),text.slice(index)];
}

export function removeUnsupportedAbsence(value,document){
  const original=String(value||"");
  const [statement,citation]=separateClaimAndCitation(original);
  if(!NEGATIVE.test(statement))return original;
  const source=selectOperativeSourceText(document).replace(/\s+/g," ");
  if(EXPLICIT_SOURCE_NEGATION.test(source))return original; // validator still checks scope
  const segments=statement.split(/;\s*/u);
  if(segments.length<2)return original; // never guess what remains of one inseparable claim
  const retained=segments.filter(part=>!NEGATIVE.test(part.trim()));
  if(retained.length===segments.length||!retained.length)return original;
  const remainder=retained.join("; ").replace(/[\s.;]+$/gu,"").trim();
  return remainder+(citation?" "+citation.trimStart():"");
}
