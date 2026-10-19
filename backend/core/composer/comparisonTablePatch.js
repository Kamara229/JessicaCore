import {locateComparisonCell} from "../validator/claim/tableCellLocator.js";

/* Patch only unverified facts into the ORIGINAL source-attributed table.
 * The model can change prose/row order/format during repair; such changes
 * must not erase already validated cells or turn a parse error into success.
 * Every patched cell is independently re-validated by the normal Validator.
 */
const norm=x=>String(x||"").replace(/\*\*/g,"").replace(/\s+/g," ").trim().toLowerCase();
function parse(line){
  const s=String(line||"").trim();
  return s.startsWith("|")&&s.endsWith("|")
    ?s.slice(1,-1).split("|").map(v=>v.trim()):null;
}
function normalizedCriterion(row){
  return norm(row).replace(/[.,:;]+$/,"");
}
function originalRows(text){
  return String(text||"").split("\n").flatMap((line,index)=>{
    const c=parse(line);
    if(!c||c.length<3||!c[0]||c.every(t=>/^:?-{2,}:?$/.test(t)))
      return [];
    return [{index,criterion:c[0],cells:c}];
  });
}
function candidateValue(text,criterion,url,urls){
  const found=locateComparisonCell({
    answer:text,sourceUrls:urls,claim:{claim:criterion,sourceUrl:url}
  });
  if(!found)return null;
  // Never substitute an unrelated criterion if a table changed shape.
  const old=normalizedCriterion(criterion),newName=normalizedCriterion(found.criterion);
  if(old!==newName)return null;
  const value=String(found.value||"").trim();
  if(!value||value.length>2300)return null;
  return value;
}
export function mergeVerifiedComparisonRepair(original,revised,locks,urls){
  if(!Array.isArray(urls)||urls.length!==2||urls.some(x=>!x))
    return {complete:false,text:original,patched:0,preserved:0};
  const lines=String(original||"").split("\n");
  const verified=new Set((Array.isArray(locks)?locks:[]).map(c=>
    norm(c.sourceUrl)+"\n"+normalizedCriterion(c.criterion)));
  let patched=0,preserved=0,eligible=0;
  for(const row of originalRows(original)){
    let changed=false;
    for(const sourceUrl of urls){
      const identity=norm(sourceUrl)+"\n"+normalizedCriterion(row.criterion);
      if(verified.has(identity)){preserved++;continue;}
      const location=locateComparisonCell({
        answer:original,sourceUrls:urls,
        claim:{claim:row.criterion,sourceUrl}
      });
      if(!location||normalizedCriterion(location.criterion)!==normalizedCriterion(row.criterion))
        continue;
      eligible++;
      const next=candidateValue(revised,row.criterion,sourceUrl,urls);
      if(!next||norm(next)===norm(row.cells[location.column]))continue;
      row.cells[location.column]=next;
      changed=true;
      patched++;
    }
    if(changed)lines[row.index]="| "+row.cells.join(" | ")+" |";
  }
  return {
    complete:eligible>0||preserved>0,
    text:lines.join("\n"),patched,preserved,eligible
  };
}
