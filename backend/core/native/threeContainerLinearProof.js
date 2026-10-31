import {parseThreeContainerLinearTask} from "./threeContainerLinearIntent.js";
import {evaluateThreeContainerLinear} from "./threeContainerLinearTool.js";
export function verifiedThreeContainerLinear(task,plan,run,answer){const input=parseThreeContainerLinearTask(task);
 if(!input)return false;
 return evaluateThreeContainerLinear(input).text===answer.text;}
