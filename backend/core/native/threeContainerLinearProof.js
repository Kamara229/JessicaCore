import {parseThreeContainerLinearTask} from "./threeContainerLinearIntent.js";
import {evaluateThreeContainerLinear} from "./threeContainerLinearTool.js";
export function verifiedThreeContainerLinear(task,plan,run,answer){const input=parseThreeContainerLinearTask(task);
 if(!input || plan?.intent!=="native_three_container_linear" || answer?.source!=="tool" || run?.success!==true)return false;
 return evaluateThreeContainerLinear(input).text===answer.text;}
