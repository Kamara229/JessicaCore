import {buildClaimEvidenceContext} from "./claimEvidenceContext.js";
import {buildOriginalSourceEvidenceContext} from "./claimEvidenceMatcher.js";
import {buildComparisonCellMatrix} from "./comparisonCellMatrix.js";
import {checkRequestedComparisonCoverage} from "./comparisonRequestedCoverage.js";
import {collectComparisonCellQuotes,planComparisonBatches} from "./comparisonCellBatcher.js";
import {verifyComparisonCellProof} from "./comparisonCellProof.js";

/* Stable facade: every asserted cell must be checked against its own
 * primary source. No validator error is a positive learning signal. */
export async function validateComparisonCells(task,taskRunResult,answer) {
  const items=(taskRunResult?.results||[])
    .filter(x=>x?.tool==="web_fetch"&&x?.success===true&&x?.data?.url);
  const matrix=buildComparisonCellMatrix(answer,items.map(x=>String(x.data.url)));
  if(!matrix)return {success:true,valid:false,shouldRetry:false,claims:[],
    reason:"Не удалось сопоставить таблицу с двумя источниками"};
  const coverage=checkRequestedComparisonCoverage(task,matrix);
  if(!coverage.valid)return {success:true,valid:false,shouldRetry:true,
    claims:[],reason:coverage.reason};
  const context=buildClaimEvidenceContext({task,answer,taskRunResult});
  if(context.sourceCount!==2||!context.sourceText)return {
    success:true,valid:false,shouldRetry:false,claims:[],
    reason:"Нет достаточных данных двух первоисточников"};
  console.log("Jessica Comparison Cell Verification:",{
    rows:matrix.totalRows,cells:matrix.checks.length,
    batches:planComparisonBatches(matrix)?.chunks.length,
    sourceChars:context.sourceChars
  });
  try {
    const batch=await collectComparisonCellQuotes({
      task,matrix,sourceText:context.sourceText
    });
    if(!batch.valid)return {success:true,valid:false,shouldRetry:false,
      claims:[],reason:batch.reason};
    const proof=verifyComparisonCellProof(batch.raw,matrix,
      buildOriginalSourceEvidenceContext(taskRunResult));
    return {success:true,valid:proof.valid,shouldRetry:!proof.valid,
      claims:proof.claims,reason:proof.reason};
  }catch(error) {
    console.warn("Jessica Comparison Cell Validator error:",error?.status||error?.name||"error");
    return {success:false,valid:false,shouldRetry:false,claims:[],
      reason:"Проверка сравнительной таблицы технически недоступна"};
  }
}
