import {groundedMitBsd2Comparison} from "./nativeMitBsd2Comparison.js";
import {groundedMitBsdComparison} from "./nativeMitBsdComparison.js";
import {groundedGplApachePatentComparison} from "./nativeGplApachePatentComparison.js";
import {groundedPatentComparison as groundedPatentComparisonBase} from "./nativePatentComparisonBase.js";
export function groundedPatentComparison(task,plan,run){
  return groundedMitBsd2Comparison(task,plan,run) ||
         groundedMitBsdComparison(task,plan,run) ||
         groundedGplApachePatentComparison(task,plan,run) ||
         groundedPatentComparisonBase(task,plan,run);
}
