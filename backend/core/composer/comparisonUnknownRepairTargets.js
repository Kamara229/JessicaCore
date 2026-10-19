import {locateComparisonCell} from "../validator/claim/tableCellLocator.js";

const UNKNOWN=/не удалось (?:подтвердить|проверить|установить)|не удалось найти|not verified|unable to verify/iu;
/* Structural retry targets; a legitimate unknown may remain unknown. */
export function comparisonUnknownRepairTargets(matrix,answer,urls){
 if(!matrix?.criteria?.length||!Array.isArray(urls)||urls.length!==2)return [];
 const targets=[];
 for(const criterion of matrix.criteria) {
  for(let source=0;source<2;source++){
   const cell=locateComparisonCell({answer,sourceUrls:urls,
     claim:{claim:criterion,sourceUrl:urls[source]}});
   if(!cell||!UNKNOWN.test(cell.value))continue;
   targets.push({criterion:cell.criterion,source,
     label:"- Строка «"+cell.criterion+"», "+(source===0?"первый":"второй")+
       " документ: проверить, можно ли заменить неопределённость подтверждённым условием."});
  }
 }
 return targets;
}
