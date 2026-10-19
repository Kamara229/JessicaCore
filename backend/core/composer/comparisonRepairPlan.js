import {buildComparisonCellMatrix} from "../validator/claim/comparisonCellMatrix.js";

/* Translate Validator's attributed claims into a constrained editing plan.
 * The model should edit only rejected cells, not rewrite an already
 * evidence-checked table. Independently re-validate the merged answer.
 */
const norm=x=>String(x||"").replace(/\*\*/g,"").replace(/\s+/g," ").trim().toLowerCase();
export function buildComparisonRepairPlan(repair) {
  if(!repair?.previousAnswer||!Array.isArray(repair?.sourceUrls)||
    repair.sourceUrls.length!==2)return "";
  const matrix=buildComparisonCellMatrix(repair.previousAnswer,repair.sourceUrls);
  if(!matrix)return "";
  const verified=new Set((repair.verifiedCells||[]).map(c=>
    norm(c.sourceUrl)+"\n"+norm(c.criterion)));
  const missing=matrix.checks.filter(c=>
    !verified.has(norm(c.sourceUrl)+"\n"+norm(c.criterion)));
  if(!missing.length)return "";
  const label=sourceIndex=>sourceIndex===0?"первого":"второго";
  return [
    "ПЛАН ТОЧЕЧНОГО ИСПРАВЛЕНИЯ (редактирование таблицы, не её пересоздание):",
    "Сохрани заголовки таблицы, названия и порядок её строк в точности. Оставь все подтверждённые ячейки без изменений.",
    "Замени только следующие НЕПОДТВЕРЖДЁННЫЕ ячейки содержательными, доказуемыми формулировками:",
    ...missing.slice(0,15).map(c=>
      "- Строка «"+c.criterion+"», столбец "+label(c.source)+" источника."),
    "Не добавляй другие юридические утверждения или дополнительные строки. Верни ту же Markdown-таблицу.",
    "Если не можешь подтвердить спорный фрагмент, используй осторожную формулировку, но не делай выводов об отсутствии нормы по молчанию источника."
  ].join("\n");
}
