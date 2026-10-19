import {matchesAttributedEvidence} from "./claimEvidenceMatcher.js";
import {isUnsupportedNegativeComparisonClaim} from "./negativeComparisonClaim.js";
import {unsupportedComparisonDuty} from "./comparisonDutyEvidence.js";
import {unsupportedComparisonClaimTopic} from "./comparisonClaimRelevance.js";

/* A model response cannot certify unlisted cells or cross-source quotes. */
function unwrap(raw) {
  const value=String(raw||"").trim().replace(/^`{3}(?:json)?\s*/i,"").replace(/`{3}$/i,"");
  const start=value.indexOf("{"),end=value.lastIndexOf("}");
  return start>=0&&end>start?value.slice(start,end+1):value;
}
function validateCell(cell,quote,sources) {
  const claim={claim:cell.criterion+": "+cell.value,evidence:quote,
    sourceUrl:cell.sourceUrl,supported:false,reason:""};
  if(!quote||quote.length<18||quote.trim().split(/\s+/).length<4)
    claim.reason="Нет достаточной подтверждающей цитаты";
  else if(!matchesAttributedEvidence(claim,sources))
    claim.reason="Цитата не найдена в указанном первоисточнике";
  else if(isUnsupportedNegativeComparisonClaim(claim,true))
    claim.reason="Отсутствие условия не доказано источником";
  else if(unsupportedComparisonDuty(claim.claim,quote))
    claim.reason="Цитата не подтверждает требование";
  else if(unsupportedComparisonClaimTopic(claim.claim,quote))
    claim.reason="Цитата не подтверждает предмет утверждения";
  else claim.supported=true;
  return claim;
}
export function verifyComparisonCellProof(raw,matrix,originalSources) {
  let parsed;
  try{parsed=JSON.parse(unwrap(raw));}catch{
    return {valid:false,claims:[],reason:"Validator вернул некорректный JSON"};
  }
  if(!Array.isArray(parsed?.c))
    return {valid:false,claims:[],reason:"Validator не указал проверяемые ячейки"};
  const byId=new Map();
  for(const entry of parsed.c) {
    if(!Array.isArray(entry)||entry.length!==2||
      !Number.isInteger(entry[0])||typeof entry[1]!=="string"||byId.has(entry[0]))
      return {valid:false,claims:[],reason:"Нарушена структура проверок ячеек"};
    byId.set(entry[0],entry[1].trim().slice(0,500));
  }
  if(byId.size!==matrix.checks.length||
    [...byId.keys()].some(id=>id<1||id>matrix.checks.length))
    return {valid:false,claims:[],
      reason:"Проверено "+byId.size+" из "+matrix.checks.length+" фактических ячеек"};
  const claims=matrix.checks.map(cell=>validateCell(cell,byId.get(cell.id),originalSources));
  const unsupported=claims.filter(claim=>!claim.supported);
  return unsupported.length
    ?{valid:false,claims,
      reason:"Не подтверждено "+unsupported.length+" ячеек: "+
        unsupported.slice(0,3).map(c=>c.claim+" ("+c.reason+")").join("; ")}
    :{valid:true,claims,
      reason:"Подтверждены цитатами все "+claims.length+" фактических ячеек"};
}
