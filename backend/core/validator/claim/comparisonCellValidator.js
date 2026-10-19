import {executeAIWithRetry} from "../../../ai/aiRetry.js";
import {validatorChat} from "../../../ai/validatorClient.js";
import {buildClaimEvidenceContext} from "./claimEvidenceContext.js";
import {buildOriginalSourceEvidenceContext} from "./claimEvidenceMatcher.js";
import {buildComparisonCellMatrix} from "./comparisonCellMatrix.js";
import {checkRequestedComparisonCoverage} from "./comparisonRequestedCoverage.js";
import {buildComparisonCellCheckPrompt,buildComparisonCellCheckRequest} from "./comparisonCellCheckPrompt.js";
import {verifyComparisonCellProof} from "./comparisonCellProof.js";

/* Facade: one evidence-backed check for EVERY asserted source cell.
 * Failed and incomplete validations cannot be saved as good Experience. */
export async function validateComparisonCells(task,taskRunResult,answer) {
  const items=(taskRunResult?.results||[])
    .filter(x=>x?.tool==="web_fetch"&&x?.success===true&&x?.data?.url);
  const urls=items.map(x=>String(x.data.url));
  const matrix=buildComparisonCellMatrix(answer,urls);
  if(!matrix)return {success:true,valid:false,shouldRetry:false,claims:[],
    reason:"Не удалось надёжно сопоставить факты таблицы с обоими источниками"};
  const coverage=checkRequestedComparisonCoverage(task,matrix);
  if(!coverage.valid)return {success:true,valid:false,shouldRetry:true,
    claims:[],reason:coverage.reason};
  const ctx=buildClaimEvidenceContext({task,answer,taskRunResult});
  if(ctx.sourceCount!==2||!ctx.sourceText)return {success:true,valid:false,
    shouldRetry:false,claims:[],reason:"Нет двух независимых первоисточников"};
  console.log("Jessica Comparison Cell Verification:",{
    rows:matrix.totalRows,cells:matrix.checks.length,sourceChars:ctx.sourceChars});
  try {
    const response=await executeAIWithRetry(
      ()=>validatorChat([
        {role:"system",content:buildComparisonCellCheckPrompt()},
        {role:"user",content:buildComparisonCellCheckRequest(task,matrix,ctx.sourceText)}
      ],{comparison:true}),{label:"Comparison Cell Validator"});
    const raw=response?.choices?.[0]?.message?.content;
    const proof=verifyComparisonCellProof(raw,matrix,
      buildOriginalSourceEvidenceContext(taskRunResult));
    return {success:true,valid:proof.valid,shouldRetry:!proof.valid,
      claims:proof.claims,reason:proof.reason};
  } catch(error) {
    console.warn("Comparison Cell Validator unavailable:",error?.status||error?.name||"error");
    return {success:false,valid:false,shouldRetry:false,claims:[],
      reason:"Проверка фактов таблицы технически недоступна"};
  }
}
