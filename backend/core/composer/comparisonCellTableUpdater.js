import {buildComparisonCellMatrix} from "../validator/claim/comparisonCellMatrix.js";
import {locateComparisonCell} from "../validator/claim/tableCellLocator.js";

// Keep the original source-column mapping. Never guess its orientation.
const normalized=x=>String(x||"").replace(/\*\*/g,"").trim();
const parse=line=>{
 const s=String(line||"").trim();
 return s.startsWith("|")&&s.endsWith("|")
   ?s.slice(1,-1).split("|").map(x=>x.trim()):null;
};
export function updateComparisonTableCells(answer,urls,transform){
 const matrix=buildComparisonCellMatrix(answer,urls);
 if(!matrix)return {text:String(answer||""),changes:0};
 const lines=String(answer||"").split("\n");
 let changes=0;
 for(const check of matrix.checks){
   const value=transform(check);
   if(normalized(value)===normalized(check.value))continue;
   const found=locateComparisonCell({answer:lines.join("\n"),sourceUrls:urls,
     claim:{claim:check.criterion,sourceUrl:check.sourceUrl}});
   if(!found||normalized(found.criterion)!==normalized(check.criterion))continue;
   const candidates=lines.flatMap((line,index)=>{
     const parts=parse(line);
     return parts&&normalized(parts[0])===normalized(check.criterion)?[index]:[];
   });
   if(candidates.length!==1)continue;
   const row=parse(lines[candidates[0]]);
   if(!row||found.column>=row.length)continue;
   row[found.column]=value;
   lines[candidates[0]]="| "+row.join(" | ")+" |";
   changes++;
 }
 return {text:lines.join("\n"),changes};
}
