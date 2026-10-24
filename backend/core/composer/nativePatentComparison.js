import {groundedGplApachePatentComparison} from "./nativeGplApachePatentComparison.js";
import {groundedPatentComparison as groundedPatentComparisonBase} from "./nativePatentComparisonBase.js";
export function groundedPatentComparison(task,plan,run){
  return groundedGplApachePatentComparison(task,plan,run) ||
         groundedPatentComparisonBase(task,plan,run);
}
