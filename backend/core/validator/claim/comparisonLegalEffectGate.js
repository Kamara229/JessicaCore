/*
 * Reject citations that mention a legal topic but not the asserted effect.
 * This is a fail-closed rejection gate, NOT a semantic proof certificate.
 * Generic across paired primary-source licensing comparisons.
 */
const retaining=/\b(?:must|shall)\s+(?:retain|reproduce|include|preserve)\b/iu;
const notification=/copyright|attribution|notice/iu;
const grant=/\b(?:grant of copyright license|hereby grants to you\b[^.]{0,220}\bcopyright license)\b/iu;
const duty=/требу|обязан|долж|нужно|необходимо|must|shall/iu;
const keep=/сохран|остав|воспроизвест|включ|retain|reproduce|preserv/iu;
const absent=/не\s+(?:содерж|предусматр|имеет)|отсутств|без\s+дополнительн|никак\p{L}*\s+дополнительн/iu;

function factualContent(value) {
  return String(value||"").replace(/\*\*/g,"").trim();
}

export function rejectsUnsupportedLegalEffect(cell,quotes) {
  const value=factualContent(cell?.value);
  const topic=factualContent(cell?.criterion);
  const citations=(Array.isArray(quotes)?quotes:[]).join(" ");
  if(!value||!citations)return true;

  // Section names, citation IDs, and language-of-authority metadata
  // cannot alone establish substantive distribution restrictions.
  if(/(?:описан|указан)[^.]{0,70}(?:раздел|пункт|текст|начал)/iu.test(value))return true;

  const notices=/уведомлен|авторств|авторск|копирайт|copyright|notice|attribut/iu.test(topic);
  if(notices){
    if(duty.test(value)&&keep.test(value)
       &&!(retaining.test(citations)&&notification.test(citations)))return true;
    // Permission to add one's own copyright is not a duty to preserve
    // upstream notices and is not itself a copyright-license grant.
    if(/(?:предоставл|грант|выда[её])[^.]{0,65}(?:лицензи|копирайт)|лицензи[^.]{0,30}(?:копирайт|авторск)/iu.test(value)
       &&!grant.test(citations))return true;
    if(/(?:исходн|source)/iu.test(value)&&/(?:бинарн|binary)/iu.test(value)){
      if(!/source\s+(?:code|form)/iu.test(citations)
         ||!/binary\s+(?:form|code)/iu.test(citations))return true;
    }
  }

  const restrictions=/огранич|restrict|торгов|trademark/iu.test(topic);
  if(restrictions){
    // "English is authoritative" is document metadata, not evidence
    // of license restrictions on software use or distribution.
    if(/(?:английск\p{L}*\s+текст|только\s+английск)/iu.test(value))return true;
    // "Without additional terms or conditions" does not prove global
    // absence of restrictions or establish absence of a provision.
    if(absent.test(value)
       &&!/\b(?:no additional restrictions|not subject to any further restrictions)\b/iu.test(citations))
      return true;
  }
  return false;
}
