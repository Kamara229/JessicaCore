import {parseIntegerProductTask} from "../../../core/native/integerMultiplicationIntent.js";

export function selectIntegerMultiplicationSkill(task,skills) {
  if (!parseIntegerProductTask(task) || !Array.isArray(skills)) return null;
  const matches=skills.filter(s=>s?.id==="integer-multiplication" &&
    s.enabled!==false && s.status==="published");
  return matches.length===1 ? matches[0] : null;
}