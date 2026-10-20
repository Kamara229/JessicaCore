import {correctOverstatedUse} from "./comparisonUseClauseRepair.js";
import {correctCopyrightGrantPolarity} from "./comparisonGrantPolarity.js";
import {correctNoticeFileClaim} from "./comparisonConditionalNoticeMeaning.js";
import {removeUnsupportedAbsence} from "./comparisonUnsupportedAbsenceRepair.js";
import {criterionType} from "../validator/claim/comparisonClauseReferences.js";
import {updateComparisonTableCells} from "./comparisonCellTableUpdater.js";
export function repairComparisonCellValue(topic,value,source){
  if(topic==="use")return correctCopyrightGrantPolarity(correctOverstatedUse(value,source),source);
  if(topic==="notices")return removeUnsupportedAbsence(correctNoticeFileClaim(value,source),source);
  return value;
}

export function repairComparisonClaimMeaning(answer,run){
 const docs=(run?.results||[]).filter(x=>x?.tool==="web_fetch"&&
   x?.success===true&&x?.data?.url&&typeof x.data?.content==="string");
 if(docs.length!==2)return {text:String(answer||""),changes:0};
 const urls=docs.map(x=>String(x.data.url));
 return updateComparisonTableCells(answer,urls,cell=>
   repairComparisonCellValue(
     criterionType(cell.criterion),cell.value,docs[cell.source].data.content
   ));
}
