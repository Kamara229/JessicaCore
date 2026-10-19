import {correctOverstatedUse} from "./comparisonUseClauseRepair.js";
import {correctNoticeFileMeaning} from "./comparisonNoticeClauseRepair.js";
export function repairComparisonCellValue(topic,value,source){
  if(topic==="use")return correctOverstatedUse(value,source);
  if(topic==="notices")return correctNoticeFileMeaning(value,source);
  return value;
}
