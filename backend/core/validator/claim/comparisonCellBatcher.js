import {executeAIWithRetry} from "../../../ai/aiRetry.js";
import {validatorChat} from "../../../ai/validatorClient.js";
import {buildComparisonCellCheckPrompt,buildComparisonCellCheckRequest} from "./comparisonCellCheckPrompt.js";
import {parseComparisonCellBatch} from "./comparisonCellQuoteParser.js";

/* Divide wide tables into bounded output groups. Reserve <=840 Groq
 * completion tokens across groups to respect small account OTPM limits.
 * A single missing group is a failed validation, never a partial success.
 */
export function planComparisonBatches(matrix) {
  const count=Math.ceil(matrix.checks.length/5);
  const maxTokens=Math.floor(840/count);
  if(!count||maxTokens<160)return null;
  const chunks=[];
  for(let at=0;at<matrix.checks.length;at+=5)chunks.push(matrix.checks.slice(at,at+5));
  return {chunks,maxTokens};
}

export async function collectComparisonCellQuotes({task,matrix,sourceText}) {
  const schedule=planComparisonBatches(matrix);
  if(!schedule)return {valid:false,reason:"Слишком много ячеек для безопасной проверки"};
  const all=new Map();
  for(const checks of schedule.chunks){
    const response=await executeAIWithRetry(
      ()=>validatorChat([
        {role:"system",content:buildComparisonCellCheckPrompt()},
        {role:"user",content:buildComparisonCellCheckRequest(task,{checks},sourceText)}
      ],{comparison:true,maxComparisonTokens:schedule.maxTokens}),
      {label:"Comparison Cell Validator"});
    const raw=response?.choices?.[0]?.message?.content||"";
    const parsed=parseComparisonCellBatch(raw,checks.map(c=>c.id));
    if(!parsed.valid)return {valid:false,reason:parsed.reason};
    for(const [id,quotes] of parsed.byId)all.set(id,quotes);
  }
  if(all.size!==matrix.checks.length)
    return {valid:false,reason:"Проверены не все ячейки сравнения"};
  const c=matrix.checks.map(x=>[x.id,all.get(x.id)]);
  return {valid:true,raw:JSON.stringify({c})};
}
