import "../../tools/integerMultiplicationTool.js";
import {parseIntegerProductTask} from "../native/integerMultiplicationIntent.js";

export function buildNativeIntegerMultiplicationPlan(task,executionExperience) {
  const request=parseIntegerProductTask(task);
  if (!request) return null;
  const selected=(executionExperience?.skills||[]).filter(s=>
    s?.id==="integer-multiplication" && s.enabled!==false && s.status==="published");
  const skill=selected.length===1?selected[0]:null;
  return {
    intent:"native_integer_multiplication",requiresTools:true,
    reasoningSummary:"Одно целочисленное произведение вычисляется без AI.",
    evidence:{mode:"none",reason:"Точная арифметика не требует внешних источников."},
    experience:{used:Boolean(skill),source:skill?"native-skill":"native-capability",
      skills:skill?[{id:skill.id,name:skill.name||""}]:[]},
    steps:[{id:"native_integer_multiply",tool:"integer_multiply",
      arguments:{a:request.a,b:request.b,showWork:request.showWork},
      experienceUsed:Boolean(skill)}],
    metadata:{plannerVersion:"native-integer-product-v1",generatedAt:new Date().toISOString()}
  };
}