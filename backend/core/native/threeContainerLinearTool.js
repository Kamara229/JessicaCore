import {registerTool} from "../../tools/toolRegistry.js";
import {solveThreeContainerLinear} from "./threeContainerLinearSolver.js";
import {formatThreeContainerLinear} from "./threeContainerLinearText.js";
export function evaluateThreeContainerLinear(input={}){
  const r=solveThreeContainerLinear(input);
  if(!r)return {success:false,text:"Недопустимые арифметические условия"};
  return {success:true,data:{...input,...r},text:formatThreeContainerLinear(input,r)};
}
registerTool({name:"three_container_linear",description:"Exact integer equation",
  arguments:{multiple:"Ratio",offset:"Difference",offsetSign:"Direction",sum:"Total"},
  execute:evaluateThreeContainerLinear});
