/*
 * Parse compact proof JSON without changing the asserted cell set.
 * Supports a legacy single quotation, or 1–3 citations per cell.
 * Missing, duplicate or invented IDs fail closed.
 */
function parseJson(raw) {
  const text=String(raw||"").trim().replace(/^\x60{3}(?:json)?\s*/i,"").replace(/\x60{3}\s*$/i,"");
  const a=text.indexOf("{"),b=text.lastIndexOf("}");
  if(a<0||b<=a)return null;
  try{return JSON.parse(text.slice(a,b+1));}catch{return null;}
}
function normalizeEntry(entry) {
  if(Array.isArray(entry)&&entry.length===2)return {id:entry[0],quotes:entry[1]};
  if(entry&&typeof entry==="object"&&!Array.isArray(entry)) {
    return {id:entry.id,quotes:entry.quotes??entry.evidence};
  }
  return null;
}
export function parseComparisonCellQuotes(raw,matrix) {
  const parsed=parseJson(raw);
  const list=parsed?.c??parsed?.claims;
  if(!Array.isArray(list))return {valid:false,reason:"Validator не указал проверки ячеек",byId:null};
  const byId=new Map();
  for(const entry of list){
    const item=normalizeEntry(entry);
    if(!item||!Number.isInteger(item.id)||item.id<1||item.id>matrix.checks.length
       ||byId.has(item.id))return {valid:false,reason:"Нарушена структура или нумерация проверок",byId:null};
    const quotes=typeof item.quotes==="string"?[item.quotes]:item.quotes;
    if(!Array.isArray(quotes)||quotes.length>3
        ||quotes.some(q=>typeof q!=="string"||q.length>500))
      return {valid:false,reason:"Некорректный формат цитат",byId:null};
    byId.set(item.id,quotes.map(q=>q.trim()).filter(Boolean));
  }
  if(byId.size!==matrix.checks.length)
    return {valid:false,reason:"Проверено "+byId.size+" из "+matrix.checks.length+" ячеек",byId:null};
  return {valid:true,byId};
}
