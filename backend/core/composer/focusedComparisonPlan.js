import {buildComparisonCellMatrix} from "../validator/claim/comparisonCellMatrix.js";
import {buildComparisonEvidenceGrid} from "./comparisonEvidenceGrid.js";
import {matchComparisonCriterion} from "./comparisonCriterionMatcher.js";

/* Prepare ONLY unverified table cells with source-specific operative text. */
const norm=x=>String(x||"").replace(/\*\*/g,"").replace(/\s+/g," ").trim().toLowerCase();
export function planFocusedComparisonRepair(task,run,repair){
  const urls=repair?.sourceUrls;
  if(!repair?.previousAnswer||!Array.isArray(urls)||urls.length!==2)return null;
  const grid=buildComparisonEvidenceGrid(task,run);
  const matrix=buildComparisonCellMatrix(repair.previousAnswer,urls);
  if(!grid||!matrix)return null;
  const frozen=new Set((repair.verifiedCells||[]).map(x=>norm(x.sourceUrl)+"::"+norm(x.criterion)));
  const candidates=matrix.checks.filter(x=>!frozen.has(norm(x.sourceUrl)+"::"+norm(x.criterion)));
  const cells=[];
  for(const item of candidates.slice(0,8)){
    const id=matchComparisonCriterion(item.criterion,grid.criteria);
    const source=grid.criteria.find(x=>x.id===id);
    if(!source)continue;
    const quotes=(source.excerpts[item.source]||[]).slice(0,3)
      .map(x=>String(x).replace(/\s+/g," ").slice(0,620));
    if(!quotes.length)continue;
    cells.push({id:item.id,source:item.source,criterion:item.criterion,
      previous:item.value,quotes});
  }
  return cells.length?{cells,urls,previousAnswer:repair.previousAnswer}:null;
}
