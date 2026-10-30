import "../../tools/integerProductAddTool.js";
import {parseIntegerProductAddTask} from "../native/integerProductAddIntent.js";

export function buildNativeIntegerProductAddPlan(task){
  const request=parseIntegerProductAddTask(task);
  if(!request)return null;
  // A composite tool is a native capability, not proof that the separate
  // integer-multiplication Experience Skill was applied.
  return {intent:"native_integer_multiply_add",requiresTools:true,
    reasoningSummary:"Точное умножение и последующее сложение целых чисел без AI.",
    evidence:{mode:"none",reason:"BigInt-арифметика проверяется локально."},
    experience:{used:false,source:"native-capability",skills:[]},
    steps:[{id:"native_integer_multiply_add",tool:"integer_multiply_add",
      arguments:request,experienceUsed:false}],
    metadata:{plannerVersion:"native-integer-product-add-v1",
      generatedAt:new Date().toISOString()}
  };
}
