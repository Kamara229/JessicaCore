import {parseIntegerProductAddTask} from "../native/integerProductAddIntent.js";
import {multiplyAndAddIntegers} from "../../tools/integerProductAddTool.js";

export function verifiedNativeIntegerProductAdd(task,plan,runResult,answerResult){
  if(plan?.intent!=="native_integer_multiply_add"||plan?.evidence?.mode!=="none"||
     !["native-integer-multiply-add-proof"].includes(answerResult?.source)||
     answerResult?.success!==true||runResult?.success!==true)return false;
  const request=parseIntegerProductAddTask(task);
  if(!request||plan?.steps?.length!==1||runResult?.results?.length!==1)return false;
  const step=plan.steps[0],actual=runResult.results[0];
  if(step?.id!=="native_integer_multiply_add"||step.tool!=="integer_multiply_add"||
     step.arguments?.a!==request.a||step.arguments?.b!==request.b||
     step.arguments?.c!==request.c||step.arguments?.showWork!==request.showWork||
     actual?.id!==step.id||actual?.tool!==step.tool||actual?.success!==true)return false;
  const expected=multiplyAndAddIntegers(request);
  return expected.success===true&&actual?.data?.product===expected.data.product&&
    actual?.data?.sum===expected.data.sum&&actual?.text===expected.text&&
    String(answerResult.text||"").trim()===expected.text.trim();
}
