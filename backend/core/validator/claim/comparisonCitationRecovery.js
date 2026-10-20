import {criterionType} from "./comparisonClauseReferences.js";

/* A mixed batch may choose a valid citation ID belonging to a different
 * source or criterion. Never use that citation. Recheck only the affected
 * cells with a source-and-criterion-specific index; the independent
 * evidence and semantic validators still decide whether the claim passes.
 * Genuine unsupported [] answers are not retried here.
 */
export async function recoverMisattributedComparisonReferences({
 task,matrix,index,sourceText,misattributedIds,runBatch,maxAttempts=2
}) {
 const targets=new Set(misattributedIds||[]);
 const restored=[];
 let attempted=0;
 for(const cell of matrix?.checks||[]) {
   if(!targets.has(cell.id)||attempted>=maxAttempts)continue;
   const refs=(index||[]).filter(ref=>
     ref.source===cell.source&&ref.criterion===criterionType(cell.criterion));
   if(!refs.length)continue;
   attempted++;
   try {
     const attempt=await runBatch({
       task,route:{mode:"indexed",checks:[cell]},index:refs,
       sourceText,maxTokens:320
     });
     if(attempt?.valid!==true)continue;
     const entry=attempt.pairs?.find(pair=>pair?.[0]===cell.id);
     if(!entry||!Array.isArray(entry[1])||!entry[1].length)continue;
     if(!entry[1].every(q=>typeof q==="string"&&q.length>17))continue;
     restored.push(entry);
   }catch(error) {
     console.warn("Jessica focused citation recheck unavailable:",{
       cellId:cell.id,reason:error?.status||error?.name||"error"
     });
   }
 }
 return {attempted,restored};
}
