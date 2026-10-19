import {executeAIWithRetry} from "../../../ai/aiRetry.js";
import {validatorChat} from "../../../ai/validatorClient.js";
import {buildComparisonCellCheckPrompt,buildComparisonCellCheckRequest} from "./comparisonCellCheckPrompt.js";
import {parseComparisonCellBatch} from "./comparisonCellQuoteParser.js";
import {
  criterionType,buildComparisonReferencePrompt,
  buildComparisonReferenceRequest,resolveComparisonClauseReferences
} from "./comparisonClauseReferences.js";

/* Divide wide tables into bounded output groups. Reserve <=840 Groq
 * completion tokens across groups to respect small account OTPM limits.
 * A single missing group is a failed validation, never a partial success.
 */
export function planComparisonBatches(matrix) {
  const count=Math.ceil(matrix.checks.length/5);
  const maxTokens=Math.floor(840/count);
  if(!count||maxTokens<160)return null;
  const groupSize=Math.ceil(matrix.checks.length/count);
  const chunks=[];
  for(let at=0;at<matrix.checks.length;at+=groupSize)
    chunks.push(matrix.checks.slice(at,at+groupSize));
  return {chunks,maxTokens};
}

export async function collectComparisonCellQuotes({task,matrix,sourceText,clauseIndex=null,fallbackIndex=null}) {
  const schedule=planComparisonBatches(matrix);
  if(!schedule)return {valid:false,reason:"Слишком много ячеек для безопасной проверки"};
  const all=new Map();
  const referenceMode=Array.isArray(clauseIndex)&&clauseIndex.length>0;
  for(const checks of schedule.chunks){
    const relevant=referenceMode?clauseIndex.filter(ref=>checks.some(cell=>
      cell.source===ref.source&&criterionType(cell.criterion)===ref.criterion)):[];
    const prompt=referenceMode?buildComparisonReferencePrompt():
      buildComparisonCellCheckPrompt();
    const request=referenceMode?
      buildComparisonReferenceRequest(task,checks,relevant):
      buildComparisonCellCheckRequest(task,{checks},sourceText);
    const maxTokens=referenceMode?Math.min(330,schedule.maxTokens):
      schedule.maxTokens;
    const response=await executeAIWithRetry(
      ()=>validatorChat([
        {role:"system",content:prompt},
        {role:"user",content:request}
      ],{comparison:true,maxComparisonTokens:maxTokens}),
      {label:"Comparison Cell Validator"});
    const raw=response?.choices?.[0]?.message?.content||"";
    if(referenceMode) {
      const parsed=resolveComparisonClauseReferences(raw,checks,relevant);
      if(!parsed.valid)return {valid:false,reason:parsed.reason};
      for(const [id,quotes] of parsed.c)all.set(id,quotes);
    } else {
      const parsed=parseComparisonCellBatch(raw,checks.map(c=>c.id));
      if(!parsed.valid)return {valid:false,reason:parsed.reason};
      for(const [id,quotes] of parsed.byId)all.set(id,quotes);
    }
  }
  // Focus a small second look on one missing proof when the initial
  // broad response could not quote an indexed operative source passage.
  if(!referenceMode&&Array.isArray(fallbackIndex)){
    const cell=matrix.checks.find(c=>(all.get(c.id)||[]).length===0&&
      fallbackIndex.some(e=>e.source===c.source&&
        e.criterion===criterionType(c.criterion)));
    if(cell){
      const refs=fallbackIndex.filter(e=>e.source===cell.source&&
        e.criterion===criterionType(cell.criterion)).slice(0,5);
      try {
        const response=await executeAIWithRetry(()=>validatorChat([
          {role:"system",content:buildComparisonReferencePrompt()},
          {role:"user",content:buildComparisonReferenceRequest(task,[cell],refs)}
        ],{comparison:true,maxComparisonTokens:170}),
        {label:"Focused Comparison Proof"});
        const parsed=resolveComparisonClauseReferences(
          response?.choices?.[0]?.message?.content||"",[cell],refs);
        if(parsed.valid&&parsed.c[0]?.[1]?.length)
          all.set(cell.id,parsed.c[0][1]);
      }catch(error){
        console.warn("Jessica focused proof unavailable:",
          error?.status||error?.name||"error");
      }
    }
  }
  if(all.size!==matrix.checks.length)
    return {valid:false,reason:"Проверены не все ячейки сравнения"};
  const c=matrix.checks.map(x=>[x.id,all.get(x.id)]);
  return {valid:true,raw:JSON.stringify({c})};
}
