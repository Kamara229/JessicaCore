import {executeAIWithRetry} from "../../../ai/aiRetry.js";
import {validatorChat} from "../../../ai/validatorClient.js";
import {buildComparisonCellCheckPrompt,buildComparisonCellCheckRequest} from "./comparisonCellCheckPrompt.js";
import {parseComparisonCellBatch} from "./comparisonCellQuoteParser.js";
import {
 criterionType,buildComparisonReferencePrompt,buildComparisonReferenceRequest,
 resolveComparisonClauseReferences
} from "./comparisonClauseReferences.js";

/* One source-attributed evidence batch. No claims are certified here;
 * citation identity and semantic proof are checked by the Validator.
 */
export async function runComparisonEvidenceBatch({
 task,route,index,sourceText,maxTokens
}) {
 const checks=route.checks,byIndex=route.mode==="indexed";
 const relevant=byIndex?index.filter(ref=>checks.some(cell=>
   cell.source===ref.source&&criterionType(cell.criterion)===ref.criterion)):[];
 const system=byIndex?buildComparisonReferencePrompt():buildComparisonCellCheckPrompt();
 const user=byIndex?buildComparisonReferenceRequest(task,checks,relevant):
   buildComparisonCellCheckRequest(task,{checks},sourceText);
 const response=await executeAIWithRetry(()=>validatorChat([
   {role:"system",content:system},{role:"user",content:user}
 ],{comparison:true,maxComparisonTokens:maxTokens}),
 {label:"Comparison Evidence "+route.mode});
 const raw=response?.choices?.[0]?.message?.content||"";
 if(byIndex) {
   const parsed=resolveComparisonClauseReferences(raw,checks,relevant);
   if(!parsed.valid)return {valid:false,reason:parsed.reason};
   return {valid:true,pairs:parsed.c,misattributedIds:parsed.misattributedIds||[]};
 }
 const parsed=parseComparisonCellBatch(raw,checks.map(x=>x.id));
 if(!parsed.valid)return {valid:false,reason:parsed.reason};
 return {valid:true,pairs:[...parsed.byId]};
}
