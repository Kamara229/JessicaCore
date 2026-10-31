import {parseThreeContainerLinearTask} from "./threeContainerLinearIntent.js";
import {evaluateThreeContainerLinear} from "./threeContainerLinearTool.js";
export function verifiedThreeContainerLinear(task,plan,run,answer){const input=parseThreeContainerLinearTask(task);
 if(!input || plan?.intent!=="native_three_container_linear" || answer?.source!=="tool" || run?.success!==true ||
 plan.steps?.[0]?.tool!=="three_container_linear" ||
 run.results?.[0]?.tool!=="three_container_linear" ||
 answer?.tool!=="three_container_linear")return false;
 const proof=evaluateThreeContainerLinear(input);
 return proof.text===answer.text && proof.text===run.results?.[0]?.text &&
 JSON.stringify(proof.data)===JSON.stringify(run.results?.[0]?.data) &&
 JSON.stringify(input)===JSON.stringify(plan.steps?.[0]?.arguments);}
