/** Exclude demonstrably incomplete date replies from later Skill revisions.
 * Keep version history intact; never retroactively declare incomplete work verified.
 */
import {missingNativeDateExplanation} from "../../nativeDateExplanationEvidence.js";

export function verifiedDateExamples(examples, program) {
  const items=Array.isArray(examples)?examples:[];
  if(program?.strategy!=="date_difference")return {examples:items,removed:0};
  const valid=items.filter(item=>!missingNativeDateExplanation({
    task:item?.task,executedTools:item?.executedTools,result:item?.result
  }));
  return {examples:valid,removed:items.length-valid.length};
}
