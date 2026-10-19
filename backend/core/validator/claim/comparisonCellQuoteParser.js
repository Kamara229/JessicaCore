/*
 * Validate cell-ID keyed evidence. Support legacy single-quote strings
 * and compact multi-quote arrays, but never invent missing IDs.
 */
function jsonObject(raw) {
  const text=String(raw||"").trim().replace(/^\x60{3}(?:json)?\s*/i,"").replace(/\x60{3}\s*$/i,"");
  const a=text.indexOf("{"),b=text.lastIndexOf("}");
  if(a<0||b<=a)return null;
  try{return JSON.parse(text.slice(a,b+1));}catch{return null;}
}
function entries(data) {
  if(Array.isArray(data))return data;
  if(data&&typeof data==="object")return Object.entries(data).map(([id,quotes])=>[Number(id),quotes]);
  return null;
}
function unpack(entry) {
  if(Array.isArray(entry)&&entry.length===2)return {id:entry[0],quotes:entry[1]};
  if(entry&&typeof entry==="object"&&!Array.isArray(entry))
    return {id:entry.id,quotes:entry.quotes??entry.evidence};
  return null;
}
export function parseComparisonCellBatch(raw,allowedIds) {
  const root=jsonObject(raw);
  const list=entries(root?.c??root?.claims);
  if(!list)return {valid:false,reason:"Validator не указал проверяемые ячейки"};
  const allowed=new Set(allowedIds);
  const byId=new Map();
  for(const entry of list) {
    const item=unpack(entry);
    if(!item||!Number.isInteger(item.id)||!allowed.has(item.id)||byId.has(item.id))
      return {valid:false,reason:"Неизвестная или повторная ячейка в проверке"};
    const rawQuotes=typeof item.quotes==="string"?[item.quotes]:item.quotes;
    if(!Array.isArray(rawQuotes)||rawQuotes.length>3||
        rawQuotes.some(q=>typeof q!=="string"||q.length>500))
      return {valid:false,reason:"Неверная структура подтверждающих цитат"};
    byId.set(item.id,rawQuotes.map(q=>q.trim()).filter(Boolean));
  }
  if(byId.size!==allowed.size)
    return {valid:false,reason:"Проверено "+byId.size+" из "+allowed.size+" фактических ячеек"};
  return {valid:true,byId};
}
export function parseComparisonCellQuotes(raw,matrix) {
  return parseComparisonCellBatch(raw,matrix.checks.map(x=>x.id));
}
