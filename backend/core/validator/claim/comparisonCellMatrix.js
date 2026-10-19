import {tableTokens} from "./tableCellLocator.js";

/* Extract every asserted fact from an actual 2-source Markdown table.
 * If a source column is ambiguous, fail rather than guessing. */
const UNKNOWN=/не удалось (?:подтвердить|проверить|установить)|не (?:подтвержден[оаы]?|установлено)|not verified|unable to verify/iu;
const ABSENCE=/не содержит|отсутств|не предусматрива|does not contain|no (?:patent|provision|requirement)/iu;
function cells(line) {
  const value=String(line||"").trim();
  return value.startsWith("|")&&value.endsWith("|")
    ?value.slice(1,-1).split("|").map(x=>x.trim()):null;
}
function separator(items) {
  return items.every(value=>/^:?-{2,}:?$/.test(value));
}
function columns(header,urls) {
  const terms=urls.map((url,index)=>{
    const other=new Set(tableTokens(urls[1-index]));
    return [...new Set(tableTokens(url))].filter(x=>!other.has(x));
  });
  if(terms.some(x=>!x.length))return null;
  const ids=terms.map(unique=>{
    const positions=header.flatMap((value,i)=>i>0 &&
      tableTokens(value).some(x=>unique.includes(x))?[i]:[]);
    return positions.length===1?positions[0]:-1;
  });
  return ids[0]>0&&ids[1]>0&&ids[0]!==ids[1]?ids:null;
}
export function buildComparisonCellMatrix(answer,urls) {
  if(!Array.isArray(urls)||urls.length!==2||urls[0]===urls[1])return null;
  const checks=[],criteria=[];
  let mapped=null,totalRows=0;
  for(const line of String(answer||"").split("\n")) {
    const parts=cells(line);
    if(!parts){mapped=null;continue;}
    if(separator(parts))continue;
    const header=columns(parts,urls);
    if(header){mapped=header;continue;}
    if(!mapped)continue;
    if(!parts[0]||mapped.some(i=>i>=parts.length))return null;
    totalRows++;
    criteria.push(parts[0]);
    for(let source=0;source<2;source++) {
      const value=parts[mapped[source]];
      if(!value)return null;
      if(UNKNOWN.test(value)&&!ABSENCE.test(value))continue;
      checks.push({id:checks.length+1,criterion:parts[0],value,
        sourceUrl:urls[source],source});
    }
  }
  return totalRows>0&&checks.length>=2&&checks.length<=18
    ?{checks,criteria,totalRows,urls}:null;
}
