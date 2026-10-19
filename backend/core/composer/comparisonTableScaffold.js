import {buildComparisonEvidenceGrid} from "./comparisonEvidenceGrid.js";
import {matchComparisonCriterion} from "./comparisonCriterionMatcher.js";
import {tableTokens} from "../validator/claim/tableCellLocator.js";

const UNKNOWN="Не удалось подтвердить по прочитанному тексту";
function parse(line){
 const s=String(line||"").trim();
 return s.startsWith("|")&&s.endsWith("|")
   ?s.slice(1,-1).split("|").map(x=>x.trim()):null;
}
function separator(parts){return parts?.every(x=>/^:?-{2,}:?$/.test(x));}
function columns(header,urls){
 const unique=urls.map((url,i)=>{
   const other=new Set(tableTokens(urls[1-i]));
   return [...new Set(tableTokens(url))].filter(x=>!other.has(x));
 });
 if(unique.some(x=>!x.length))return null;
 const cols=unique.map(words=>{
   const matches=header.flatMap((cell,i)=>i>0&&
     tableTokens(cell).some(w=>words.includes(w))?[i]:[]);
   return matches.length===1?matches[0]:-1;
 });
 return cols[0]>0&&cols[1]>0&&cols[0]!==cols[1]?cols:null;
}
export function scaffoldComparisonTable(task,run,answer){
 const grid=buildComparisonEvidenceGrid(task,run);
 if(!grid||grid.sources.length!==2||!grid.criteria.length)
   return {text:String(answer||""),applied:false,reason:"no-grid"};
 const lines=String(answer||"").split("\n");
 const urls=grid.sources.map(x=>x.url);
 const headers=lines.flatMap((line,i)=>{
   const parts=parse(line);
   const cols=parts&&columns(parts,urls);
   return cols?[{index:i,parts,cols}]:[];
 });
 if(headers.length!==1)return {text:String(answer||""),applied:false,reason:"ambiguous-header"};
 const header=headers[0];
 const byId=new Map(), seen=new Set();
 let end=header.index+1;
 for(;end<lines.length;end++){
   const cells=parse(lines[end]);
   if(!cells)break;
   if(separator(cells))continue;
   const id=matchComparisonCriterion(cells[0],grid.criteria);
   if(!id)continue;
   if(seen.has(id))return {text:String(answer||""),applied:false,reason:"duplicate-criterion"};
   if(header.cols.some(col=>col>=cells.length))continue;
   seen.add(id);
   byId.set(id,header.cols.map(col=>cells[col]||UNKNOWN));
 }
 const missing=grid.criteria.filter(c=>!byId.has(c.id));
 if(!missing.length)return {text:String(answer||""),applied:false,reason:"complete"};
 const normalized=[
  "| Критерий | "+header.parts[header.cols[0]]+" | "+
      header.parts[header.cols[1]]+" |",
  "| --- | --- | --- |",
  ...grid.criteria.map(c=>{
    const values=byId.get(c.id)||[UNKNOWN,UNKNOWN];
    return "| "+c.title+" | "+values[0]+" | "+values[1]+" |";
  })
 ];
 lines.splice(header.index,end-header.index,...normalized);
 return {text:lines.join("\n"),applied:true,missing:missing.map(x=>x.id),
   preserved:byId.size};
}
