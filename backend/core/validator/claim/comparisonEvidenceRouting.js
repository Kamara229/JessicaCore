import {criterionType} from "./comparisonClauseReferences.js";

/* Per-cell routing instead of all-or-nothing indexed validation.
 * Clauses can be absent for one criterion without disabling
 * reliable source-indexed evidence for all other criteria.
 */
export function planComparisonEvidenceRoutes(matrix,index) {
 const claimed=Array.isArray(matrix?.checks)?matrix.checks:[];
 if(!claimed.length)return [];
 const refs=Array.isArray(index)?index:[];
 const indexed=[],verbatim=[];
 for(const cell of claimed){
   const supportedByIndex=refs.some(ref=>
     ref.source===cell.source&&ref.criterion===criterionType(cell.criterion));
   (supportedByIndex?indexed:verbatim).push(cell);
 }
 const groups=[];
 for(const [mode,items] of [["indexed",indexed],["verbatim",verbatim]]){
   for(let i=0;i<items.length;i+=4)
     groups.push({mode,checks:items.slice(i,i+4)});
 }
 return groups;
}
