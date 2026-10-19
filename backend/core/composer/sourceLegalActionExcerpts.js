import {selectOperativeSourceText} from "../validator/claim/operativeSourceScope.js";

/*
 * Prefer operative permissions, duties and prohibitions over headings,
 * appendix instructions and notes on the language of a document.
 * Returned strings are primary-source snippets, NOT legal conclusions.
 * Patterns describe reusable legal actions; they do not name licenses.
 */
const RULES={
  use:[
    /\bhereby\s+grants\b.{0,200}\bcopyright\s+license\b/iu,
    /\bredistribution\s+and\s+use\s+in\s+source\s+and\s+binary\s+forms\b/iu,
    /\bpermission\s+is\s+hereby\s+granted\b/iu,
    /\byou\s+may\s+reproduce\s+and\s+distribute\b/iu
  ],
  distribution:[
    /\byou\s+may\s+reproduce\s+and\s+distribute\s+copies\b/iu,
    /\bredistribution\s+and\s+use\s+in\s+source\s+and\s+binary\s+forms\b/iu,
    /\bredistributions?\s+of\s+source\s+code\s+must\s+retain\b/iu,
    /\bredistributions?\s+in\s+binary\s+form\s+must\s+reproduce\b/iu
  ],
  restrictions:[
    /\b(?:this\s+license\s+)?does\s+not\s+grant\s+permission\s+to\s+use\s+the\s+trade\s+names\b/iu,
    /\bneither\s+the\s+name\s+of\s+the\s+copyright\s+holder\b/iu,
    /\b(?:may\s+not|must\s+not|shall\s+not)\s+use\b.{0,80}\b(?:name|trademark|trade\s+name)\b/iu,
    /\bwithout\s+(?:specific\s+)?prior\s+written\s+permission\b/iu
  ]
};
function window(text,offset) {
  let start=Math.max(0,offset-100),end=Math.min(text.length,offset+455);
  if(start>0){const i=text.indexOf(" ",start);if(i>=0&&i<offset)start=i+1;}
  if(end<text.length){const i=text.lastIndexOf(" ",end);if(i>offset+130)end=i;}
  return text.slice(start,end).trim();
}
export function extractLegalActionExcerpts(document,criterion) {
  const rules=RULES[String(criterion||"")];
  if(!rules)return [];
  const text=selectOperativeSourceText(document).replace(/\s+/g," ").slice(0,50000);
  const found=[];
  for(const rule of rules) {
    const match=rule.exec(text);
    if(!match)continue;
    if(found.some(item=>Math.abs(item.offset-match.index)<180))continue;
    found.push({offset:match.index,quote:window(text,match.index)});
    if(found.length===2)break;
  }
  return found.map(x=>x.quote).filter(x=>x.length>45);
}
