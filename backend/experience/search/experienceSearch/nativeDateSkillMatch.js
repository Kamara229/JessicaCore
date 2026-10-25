import {extractExplicitDateRange} from "../../../core/native/explicitDateRange.js";
import {extractNativeDates} from "../../../core/native/dateUtils.js";
import {resolveSkillExecutionProgram} from "../../execution/executionProgram.js";

export function selectDateSkill(task, skills) {
  if (typeof task !== "string" || !Array.isArray(skills)) return null;
  const text=task.toLowerCase();
  if (!/(дн|day)/iu.test(text) ||
      !/(сколько|вычисл|рассч|количеств|прошло|определ|calculate|difference)/iu.test(text)) return null;
  if (!extractExplicitDateRange(task) && extractNativeDates(task).length<2) return null;
  const found=skills.filter(skill => skill && skill.enabled!==false &&
      (!skill.status || skill.status==="published") &&
      resolveSkillExecutionProgram(skill)?.strategy==="date_difference");
  return found.length===1 ? found[0] : null;
}
