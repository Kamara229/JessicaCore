import "../native/threeContainerLinearTool.js";
import {parseThreeContainerLinearTask} from "../native/threeContainerLinearIntent.js";
export function buildNativeThreeContainerLinearPlan(task){
 const p=parseThreeContainerLinearTask(task);if(!p)return null;
 return {intent:"native_three_container_linear",requiresTools:true,
 reasoningSummary:"Exact integer equation",evidence:{mode:"none"},
 steps:[{id:"native_three_container_linear",tool:"three_container_linear",arguments:p}],
 metadata:{plannerVersion:"native-three-container-v1"}};
}
