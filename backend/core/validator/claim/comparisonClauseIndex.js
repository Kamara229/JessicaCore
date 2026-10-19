import {selectOperativeSourceText} from "./operativeSourceScope.js";

/* Index contiguous operative provisions in retrieved source text.
 * The model selects clause IDs instead of copying 200-word quotations.
 * Quote text is always reconstructed from the ORIGINAL source.
 * This is generic to English-language rights/conditions documents.
 */
const ANCHORS=[
  ["use",/\bhereby grants to you\b.{0,180}\bcopyright license\b/giu],
  ["use",/\bredistribution and use in source and binary forms\b/giu],
  ["distribution",/\byou may reproduce and distribute copies\b/giu],
  ["distribution",/\bredistributions? of source code must retain\b/giu],
  ["distribution",/\bredistributions? in binary form must reproduce\b/giu],
  ["notices",/\byou must retain\b/giu],
  ["notices",/\byou must give\b/giu],
  ["notices",/\bif the work includes a\b.{0,60}\bnotice\b/giu],
  ["notices",/\bderivative works\b.{0,140}\bmust include\b/giu],
  ["notices",/\bredistributions? of source code must retain\b/giu],
  ["notices",/\bredistributions? in binary form must reproduce\b/giu],
  ["patents",/\bhereby grants to you\b.{0,230}\bpatent license\b/giu],
  ["patents",/\bif you institute patent litigation\b/giu],
  ["patents",/\bpatent licenses granted\b.{0,160}\bshall terminate\b/giu],
  ["restrictions",/\bdoes not grant permission to use the trade names\b/giu],
  ["restrictions",/\bneither the name of the copyright holder\b/giu],
  ["restrictions",/\bwithout specific prior written permission\b/giu],
  ["warranty",/\bprovided by the copyright holders and contributors\b/giu],
  ["warranty",/\bprovided on an as is basis\b/giu]
];
const WINDOW_WORDS=56;
function quoteAt(source,start) {
  const tail=source.slice(start);
  const words=tail.split(/\s+/).filter(Boolean);
  return words.slice(0,WINDOW_WORDS).join(" ").trim();
}
export function buildComparisonClauseIndex(run) {
  const found=(run?.results||[]).filter(x=>
    x?.tool==="web_fetch"&&x?.success===true&&x?.data?.url&&
    typeof x.data.content==="string"&&x.data.content.length>100);
  if(found.length!==2)return null;
  const citations=[];
  for(let source=0;source<found.length;source++) {
    const text=selectOperativeSourceText(found[source].data.content)
      .replace(/\s+/g," ").trim();
    for(const [criterion,regex] of ANCHORS){
      regex.lastIndex=0;
      let count=0;
      for(const match of text.matchAll(regex)){
        if(count>=2)break;
        const quote=quoteAt(text,match.index);
        if(quote.split(" ").length<5)continue;
        if(citations.some(x=>x.source===source&&x.criterion===criterion&&
          x.quote===quote))continue;
        citations.push({id:"E"+(citations.length+1),source,
          criterion,url:String(found[source].data.url),quote});
        count++;
      }
    }
  }
  return citations.length?citations:null;
}
export function formatComparisonClauseIndex(index) {
  return index.map(x=>x.id+" SOURCE "+(x.source+1)+
    " "+x.criterion+": "+x.quote).join("\n");
}
