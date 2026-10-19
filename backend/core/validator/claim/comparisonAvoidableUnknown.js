import {locateComparisonCell} from "./tableCellLocator.js";
import {selectOperativeSourceText} from "./operativeSourceScope.js";
import {containsExplicitNoticeDuty} from "./comparisonExplicitDuties.js";
import {indexExplicitComparisonTopics,hasExplicitComparisonTopic} from "./comparisonUnknownProofIndex.js";

/*
 * An unknown table cell is sometimes appropriate, but not when the
 * retrieved source contains an explicit, directly relevant provision.
 * This only requests a better answer; it NEVER certifies a claim.
 */
const UNKNOWN=/не удалось (?:подтвердить|проверить|установить)|не удалось найти|не (?:подтвержден|установлено)|not verified|unable to verify/iu;
const ACTIONS=[
  {id:"use",claim:"использование",asked:/использ|модифиц|изменен|\buse\b|modif/iu,
    proof:/\b(?:hereby grants to you|redistribution and use in source and binary forms|permission is hereby granted)\b/iu},
  {id:"restrictions",claim:"ограничения",asked:/огранич|товарн|trademark|restrict/iu,
    proof:/\b(?:does not grant permission to use|neither the name of the copyright holder|without specific prior written permission)\b/iu},
  {id:"notices",claim:"сохранение уведомлений",asked:/уведомлен|авторск|copyright|notic/iu,
    proof:/\bmust\s+(?:retain|reproduce|include)\b[^.]{0,200}\b(?:copyright|attribution|notices?)\b/iu},
  {id:"distribution",claim:"распространение",asked:/распростран|distribut/iu,
    proof:/\b(?:may|permitted)\b[^.]{0,90}\bdistribut|\bredistribution\s+and\s+use\b/iu},
  {id:"patents",claim:"патентные условия",asked:/патент|patent/iu,
    proof:/\b(?:hereby grants|grant of patent license)\b[^.]{0,180}\bpatent\b/iu}
];
export function findAvoidableUnknownComparisonCell(task,answer,sources) {
  if(!Array.isArray(sources)||sources.length!==2)return null;
  const urls=sources.map(x=>String(x?.data?.url||""));
  if(urls.some(x=>!x))return null;
  const positive=indexExplicitComparisonTopics(sources);
  for(const rule of ACTIONS){
    if(!rule.asked.test(String(task||"")))continue;
    for(let index=0;index<2;index++){
      const cell=locateComparisonCell({
        answer,sourceUrls:urls,claim:{claim:rule.claim,sourceUrl:urls[index]}
      });
      if(!cell||!UNKNOWN.test(cell.value))continue;
      const primary=selectOperativeSourceText(sources[index]?.data?.content||"").replace(/\s+/g," ");
      // A Markdown-formatted source can hide a duty from the compact
      // regex. Cross-check the normalized operative text before treating
      // a plainly stated requirement as unverified.
      const directlyConfirmed=rule.claim==="сохранение уведомлений" &&
        containsExplicitNoticeDuty(sources[index]?.data?.content);
      if(rule.proof.test(primary)||directlyConfirmed||
        hasExplicitComparisonTopic(positive,index,rule.id))return {
        criterion:cell.criterion,sourceUrl:urls[index],
        reason:"По критерию «"+cell.criterion+"» указано «не удалось подтвердить», хотя первоисточник содержит явно относящееся к нему положение. Пересмотри соответствующую ячейку по тексту источника."
      };
    }
  }
  return null;
}
