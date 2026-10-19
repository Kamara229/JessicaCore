import {locateComparisonCell} from "../validator/claim/tableCellLocator.js";

/* Apply only the fields explicitly included in the repair plan. */
function parseJson(raw){
 const text=String(raw||"").trim().replace(/^`{3}(?:json)?\s*/i,"").replace(/`{3}\s*$/,"");
 try{return JSON.parse(text);}catch{return null;}
}
const norm=s=>String(s||"").replace(/\*\*/g,"").replace(/\s+/g," ").trim();
function row(line){
 const v=String(line||"").trim();
 return v.startsWith("|")&&v.endsWith("|")?v.slice(1,-1).split("|").map(x=>x.trim()):null;
}
export function patchFocusedComparison(plan,raw){
 const root=parseJson(raw);
 const claims=root?.c;
 if(!Array.isArray(claims)||!claims.length||claims.length>plan.cells.length)
   return {ok:false};
 const allowed=new Map(plan.cells.map(x=>[x.id,x]));
 const lines=plan.previousAnswer.split("\n");
 const seen=new Set();
 let changed=0;
 for(const entry of claims){
   if(!Array.isArray(entry)||entry.length!==2)return {ok:false};
   const [id,updated]=entry,cell=allowed.get(id);
   if(!cell||seen.has(id)||typeof updated!=="string")return {ok:false};
   if(updated.length<15||updated.length>1100||/[|\r\n]/.test(updated))return {ok:false};
   seen.add(id);
   const located=locateComparisonCell({
     answer:plan.previousAnswer,sourceUrls:plan.urls,
     claim:{claim:cell.criterion,sourceUrl:plan.urls[cell.source]}
   });
   if(!located||norm(located.criterion)!==norm(cell.criterion))return {ok:false};
   const rows=lines.flatMap((line,index)=>{
     const r=row(line);return r&&norm(r[0])===norm(cell.criterion)?[index]:[];
   });
   if(rows.length!==1)return {ok:false};
   const parts=row(lines[rows[0]]);
   if(!parts||located.column>=parts.length)return {ok:false};
   if(norm(parts[located.column])!==norm(updated)){
     parts[located.column]=updated.trim();
     lines[rows[0]]="| "+parts.join(" | ")+" |";
     changed++;
   }
 }
 return {ok:changed>0,text:lines.join("\n"),changed};
}
