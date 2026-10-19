import {getApprovalExperience} from "../approval/approvalMetrics.js";

/*
 * Evidence written to long-term Skills must show source provenance.
 * A formally "verified" comparison without its requested official
 * citations may be a useful draft, but is not a trusted learning example.
 * This is a generic cross-document rule, not a license-specific answer.
 */
function needsOfficialComparativeCitations(task) {
  const t=String(task||"");
  return /сравн|сопостав|compar(?:e|ison)/iu.test(t) &&
    /официальн|official/iu.test(t) &&
    /источник|ссылк|подтверд|цитат|source|citat|referenc/iu.test(t);
}
function uniqueCitedUrls(value) {
  const raw=String(value||"");
  const urls=[...raw.matchAll(/https?:\/\/[^\s<>\])}"'«»]+/giu)]
    .map(m=>m[0].replace(/[.,;]+$/,"").toLowerCase());
  return new Set(urls).size;
}
export function validateGroundedComparisonLearning(proposal) {
  const experience=getApprovalExperience(proposal);
  const examples=Array.isArray(experience?.examples)?experience.examples:[];
  for(const example of examples) {
    if(!needsOfficialComparativeCitations(example?.task))continue;
    const output=example?.result;
    if(example?.success!==true||output?.verified!==true ||
        output?.success!==true||output?.status!=="COMPLETED")return {
      passed:false,
      reason:"Неподтверждённый результат не может формировать Skill сравнения"
    };
    if(uniqueCitedUrls(output?.text)<2)return {
      passed:false,
      reason:"Результат сравнения не содержит две прямые ссылки на первоисточники"
    };
  }
  return {passed:true,reason:"Source-grounded learning examples have required references"};
}
