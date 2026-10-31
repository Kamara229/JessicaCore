import "../../tools/bridgeCrossingTool.js";
import {parseBridgeCrossingTask} from "../native/bridgeCrossingIntent.js";
export function buildNativeBridgeCrossingPlan(task) {
  const p=parseBridgeCrossingTask(task);if(!p)return null;
  return {intent:"native_bridge_crossing",requiresTools:true,
    reasoningSummary:"Кратчайшая переправа определяется полным перебором разрешённых состояний.",
    evidence:{mode:"none",reason:"Детерминированный расчёт полного графа допустимых переходов."},
    experience:{used:false,source:"native-capability",skills:[]},
    steps:[{id:"native_bridge_crossing",tool:"bridge_crossing_optimal",
      arguments:{times:p.times,explain:p.explain},experienceUsed:false}],
    metadata:{plannerVersion:"native-bridge-v1",generatedAt:new Date().toISOString()}};
}
