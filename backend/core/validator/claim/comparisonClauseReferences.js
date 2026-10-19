import {parseComparisonCellBatch} from "./comparisonCellQuoteParser.js";
import {formatComparisonClauseIndex} from "./comparisonClauseIndex.js";

/* Compact validator contract: IDs refer to immutable verbatim source
 * spans. Never accept citations belonging to another document.
 * Every requested cell remains mandatory, even when unsupported ([]).
 */
export function criterionType(value) {
  const s=String(value||"");
  if(/патент|patent/iu.test(s))return "patents";
  if(/уведомлен|авторск|копирайт|copyright|notice|attribut/iu.test(s))return "notices";
  if(/распростран|distribut/iu.test(s))return "distribution";
  if(/огранич|товарн|торгов|trademark|restrict|endorse/iu.test(s))return "restrictions";
  if(/использован|модифиц|изменен|изменения|\buse\b|modif/iu.test(s))return "use";
  if(/гарант|ответственн|warrant|liabilit/iu.test(s))return "warranty";
  return "";
}
export function canUseComparisonClauseIndex(matrix,index) {
  if(!index||!matrix?.checks?.length)return false;
  return matrix.checks.every(c=>index.some(p=>
    p.source===c.source && p.criterion===criterionType(c.criterion)));
}
export function buildComparisonReferencePrompt() {
  return [
    "Проверяй фактические ячейки таблицы по НУМЕРОВАННЫМ выдержкам первоисточников.",
    "Каждая выдержка E# принадлежит конкретному SOURCE. Нельзя ссылаться на другой SOURCE.",
    "Верни только JSON: {\"c\":[[1,[\"E2\",\"E4\"]],[2,[]]]}.",
    "Внутри каждого ID перечисли 1-3 E#; не переписывай сами цитаты.",
    "Если хотя бы часть ячейки не подтверждена представленными выдержками — верни [] для её ID.",
    "Не подтверждай отсутствие условий молчанием источника.",
    "Для многосоставных утверждений нужны выдержки, подтверждающие КАЖДУЮ часть.",
    "Для патентного гранта и прекращения права нужны разные положения.",
    "Для обязанностей выбирай must/shall/retain/reproduce, не простое may.",
    "Все ID в ответе должны присутствовать ровно один раз."
  ].join("\n");
}
export function buildComparisonReferenceRequest(task,checks,index) {
  return [
    "ЗАДАНИЕ: "+String(task||""),"",
    "ПРОВЕРЯЕМЫЕ ЯЧЕЙКИ:",
    ...checks.map(c=>c.id+". SOURCE "+(c.source+1)+
      "; КРИТЕРИЙ "+c.criterion+"; ТЕКСТ "+c.value),
    "","ВЫДЕРЖКИ ПЕРВОИСТОЧНИКОВ:",formatComparisonClauseIndex(index)
  ].join("\n");
}
export function resolveComparisonClauseReferences(raw,checks,index) {
  const parsed=parseComparisonCellBatch(raw,checks.map(x=>x.id));
  if(!parsed.valid)return {valid:false,reason:parsed.reason};
  const byRef=new Map(index.map(item=>[item.id,item]));
  const out=[];
  for(const cell of checks){
    const refs=parsed.byId.get(cell.id)||[];
    const quotes=[];
    const seen=new Set();
    for(const ref of refs){
      if(!/^E\d+$/.test(ref)||seen.has(ref))
        return {valid:false,reason:"Повторный или некорректный ID выдержки"};
      const excerpt=byRef.get(ref);
      if(!excerpt || excerpt.source!==cell.source ||
        excerpt.criterion!==criterionType(cell.criterion))
        return {valid:false,reason:"Источник выдержки не соответствует ячейке"};
      seen.add(ref);
      quotes.push(excerpt.quote);
    }
    out.push([cell.id,quotes]);
  }
  return {valid:true,c:out};
}
