import {selectOperativeSourceText} from "./operativeSourceScope.js";

/* Prefer two independent patent provisions: grant and termination.
 * These are source excerpts, not a statement that every license has them.
 * The absence of a match is NEVER proof that rights are absent. */
function windowAt(source,at,length=430){
  const start=Math.max(0,at-90);
  const end=Math.min(source.length,at+length);
  let a=start,b=end;
  if(a>0){const space=source.indexOf(" ",a);if(space>=0&&space<at)a=space+1;}
  if(b<source.length){const space=source.lastIndexOf(" ",b);if(space>at+90)b=space;}
  return source.slice(a,b).trim();
}
export function extractPatentClauseExcerpts(document,task) {
  if(!/патент|patent/iu.test(String(task||"")))return [];
  const text=selectOperativeSourceText(document).replace(/\s+/g," ").slice(0,30000);
  const patterns=[
    /grant of patent license|hereby grants.{0,100}patent license/iu,
    /if you institute patent litigation|patent licenses granted.{0,110}shall terminate/iu
  ];
  const out=[];
  for(const pattern of patterns){
    const match=pattern.exec(text);
    if(!match)continue;
    const value=windowAt(text,match.index);
    if(value.length>35&&!out.includes(value))out.push(value);
  }
  return out;
}
