import {buildComparisonCellMatrix} from "../validator/claim/comparisonCellMatrix.js";
import {locateComparisonCell} from "../validator/claim/tableCellLocator.js";

/*
 * Surgical comparison repair: do NOT let the Composer rewrite facts that
 * the independent Validator already checked successfully.
 * Only an unsupported cell may change. This is not a proof mechanism.
 */
const norm=s=>String(s||"").replace(/\*\*/g,"").replace(/\s+/g," ").trim();
const parse=s=>{
  const v=String(s||"").trim();
  return v.startsWith("|")&&v.endsWith("|")
    ?v.slice(1,-1).split("|").map(x=>x.trim()):null;
};
export function freezeVerifiedComparisonCells(answer,claims,urls) {
  const matrix=buildComparisonCellMatrix(answer,urls);
  if(!matrix||!Array.isArray(claims))return [];
  const verified=new Set(claims.filter(c=>c?.supported===true)
    .map(c=>String(c.sourceUrl||"")+"\n"+norm(c.claim)));
  return matrix.checks.filter(c=>
    verified.has(c.sourceUrl+"\n"+norm(c.criterion+": "+c.value))
  ).map(c=>({criterion:c.criterion,sourceUrl:c.sourceUrl,value:c.value}));
}

export function restoreVerifiedComparisonCells(answer,locks,urls) {
  if(!Array.isArray(locks)||locks.length===0)
    return {complete:true,text:answer,restored:0};
  const lines=String(answer||"").split("\n");
  let restored=0;
  for(const lock of locks) {
    const location=locateComparisonCell({
      answer:lines.join("\n"),sourceUrls:urls,
      claim:{claim:lock.criterion,sourceUrl:lock.sourceUrl}
    });
    if(!location||norm(location.criterion)!==norm(lock.criterion))
      return {complete:false,text:answer,restored};
    const rows=lines.flatMap((line,i)=>{
      const cells=parse(line);
      return cells&&norm(cells[0])===norm(lock.criterion)?[i]:[];
    });
    if(rows.length!==1)return {complete:false,text:answer,restored};
    const i=rows[0],cells=parse(lines[i]);
    if(!cells||location.column>=cells.length)return {complete:false,text:answer,restored};
    cells[location.column]=lock.value;
    lines[i]="| "+cells.join(" | ")+" |";
    restored++;
  }
  return {complete:true,text:lines.join("\n"),restored};
}
