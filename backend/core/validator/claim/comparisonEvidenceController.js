import {planComparisonEvidenceRoutes} from "./comparisonEvidenceRouting.js";
import {runComparisonEvidenceBatch} from "./comparisonEvidenceBatchRun.js";

/* Small facade: select evidence mode per cell, then require complete,
 * attributed proof. No result is certified here; legal semantics are
 * checked by the independent Validator after quote matching.
 */
export function planComparisonBatches(matrix,index=[]) {
 const routes=planComparisonEvidenceRoutes(matrix,index);
 if(!routes.length||routes.length>5)return null;
 const maxTokens=Math.floor(840/routes.length);
 return maxTokens>=160?{routes,maxTokens,chunks:routes.map(r=>r.checks)}:null;
}
export async function collectComparisonCellQuotes({
 task,matrix,sourceText,clauseIndex=null,fallbackIndex=null
}) {
 const index=clauseIndex||fallbackIndex||[];
 const plan=planComparisonBatches(matrix,index);
 if(!plan)return {valid:false,reason:"Слишком много ячеек для проверки"};
 const all=new Map();
 console.log("Jessica Comparison Evidence Routing:",plan.routes.map(x=>({
   mode:x.mode,cells:x.checks.length
 })));
 for(const route of plan.routes){
   const result=await runComparisonEvidenceBatch({
     task,route,index,sourceText,
     maxTokens:Math.min(plan.maxTokens,route.mode==="indexed"?330:420)
   });
   if(!result.valid)return {valid:false,reason:result.reason};
   for(const [id,quotes] of result.pairs)all.set(id,quotes);
 }
 if(all.size!==matrix.checks.length)
   return {valid:false,reason:"Подтверждения получены не для всех ячеек"};
 return {valid:true,raw:JSON.stringify({c:matrix.checks.map(x=>[
   x.id,all.get(x.id)
 ])})};
}
