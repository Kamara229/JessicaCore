import {parseComparisonCellQuotes} from "./comparisonCellQuoteParser.js";
import {verifyComparisonCellEvidence} from "./comparisonCellEvidence.js";

/* Stable facade: parse exactly the cells requested, verify each citation,
 * and fail closed on missing or unsupported facts. */
export function verifyComparisonCellProof(raw,matrix,originalSources) {
  const parsed=parseComparisonCellQuotes(raw,matrix);
  if(!parsed.valid)return {valid:false,claims:[],reason:parsed.reason};
  const claims=matrix.checks.map(cell=>
    verifyComparisonCellEvidence(cell,parsed.byId.get(cell.id),originalSources));
  const bad=claims.filter(c=>!c.supported);
  if(bad.length)return {valid:false,claims,reason:
    "Не подтверждено "+bad.length+" ячеек: "+
      bad.slice(0,3).map(c=>c.claim+" ("+c.reason+")").join("; ")};
  return {valid:true,claims,
    reason:"Все "+claims.length+" фактических ячеек подтверждены цитатами"};
}
