import {registerTool} from "./toolRegistry.js";
import {shortestBridgeCrossing} from "../core/native/bridgeCrossingShortestPath.js";
import {formatBridgeCrossingProof} from "../core/native/bridgeCrossingProofText.js";
export function solveBridgeCrossingTool({times,explain=true}={}) {
  const solved=shortestBridgeCrossing(times);
  if(!solved)return {success:false,text:"Недопустимые времена переходов"};
  return {success:true,data:{times,minutes:solved.minutes,reachable:solved.reachable,
    route:solved.route,distances:solved.distances},
    text:formatBridgeCrossingProof(times,solved,explain)};
}
registerTool({name:"bridge_crossing_optimal",
  description:"Exact optimal four-person flashlight bridge crossing by finite-state shortest path",
  arguments:{times:"Four crossing times in minutes",explain:"Include exhaustive proof table"},
  execute:solveBridgeCrossingTool});
