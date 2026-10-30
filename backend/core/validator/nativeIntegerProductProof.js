import {parseIntegerProductTask} from "../native/integerMultiplicationIntent.js";
import {multiplyIntegers} from "../../tools/integerMultiplicationTool.js";

// Validate the precise executed product, not merely a plausible-looking answer.
// Generic native-fallback replies and non-arithmetic tool output remain unverified.
export function verifiedNativeIntegerProduct(task,plan,runResult,answerResult) {
  if (plan?.intent!=="native_integer_multiplication" ||
      plan?.evidence?.mode!=="none" ||
      !["native-fallback","tool"].includes(answerResult?.source) ||
      answerResult?.success!==true || runResult?.success!==true) return false;
  const request=parseIntegerProductTask(task);
  if (!request || plan?.steps?.length!==1 ||
      runResult?.results?.length!==1) return false;
  const step=plan.steps[0], actual=runResult.results[0];
  if (step?.id!=="native_integer_multiply" || step.tool!=="integer_multiply" ||
      step.arguments?.a!==request.a || step.arguments?.b!==request.b ||
      step.arguments?.showWork!==request.showWork ||
      actual?.id!==step.id || actual?.tool!==step.tool ||
      actual?.success!==true) return false;
  const expected=multiplyIntegers(request);
  if (expected.success!==true || actual?.data?.product!==expected.data.product ||
      actual.text!==expected.text) return false;
  return String(answerResult.text||"").trim()===expected.text.trim();
}
