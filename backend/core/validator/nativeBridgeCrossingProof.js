import {parseBridgeCrossingTask} from "../native/bridgeCrossingIntent.js";
import {solveBridgeCrossingTool} from "../../tools/bridgeCrossingTool.js";
/** Recompute from the original user request; reject forged or altered tool results. */
export function verifiedNativeBridgeCrossing(task,plan,runResult,answerResult){
  if(plan?.intent!=="native_bridge_crossing" || plan?.evidence?.mode!=="none" ||
     runResult?.success!==true || answerResult?.success!==true ||
     answerResult?.tool!=="bridge_crossing_optimal" || answerResult?.source!=="tool" ||
     plan.steps?.length!==1 || runResult.results?.length!==1)return false;
  const p=parseBridgeCrossingTask(task),step=plan.steps[0],actual=runResult.results[0];
  if(!p || step?.id!=="native_bridge_crossing" || step.tool!=="bridge_crossing_optimal" ||
     JSON.stringify(step.arguments)!==JSON.stringify(p) ||
     actual?.id!==step.id || actual.tool!==step.tool || actual.success!==true)return false;
  const expected=solveBridgeCrossingTool(p);
  return expected.success===true &&
    JSON.stringify(actual.data)===JSON.stringify(expected.data) &&
    actual.text===expected.text && answerResult.text===expected.text;
}
